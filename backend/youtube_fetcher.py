import os
import time
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Ładujemy klucz z pliku .env
load_dotenv()
API_KEY = os.getenv('key')

# Inicjalizacja klienta YouTube API
youtube = build('youtube', 'v3', developerKey=API_KEY)


def fetch_comments(video_id, max_results=1000, order="time"):
    """
    Pobiera komentarze dla zadanego ID filmu na YouTube z paginacją.

    Parametry:
        video_id    - ID filmu (znaki po "v=" w linku YouTube)
        max_results - maksymalna łączna liczba komentarzy do pobrania (domyślnie 500)
        order       - "time" (chronologicznie) lub "relevance" (najpopularniejsze)

    Zwraca DataFrame z kolumnami:
        author, text, published_at, likes, reply_count
    """
    comments_list = []
    next_page_token = None
    # API zwraca max 100 komentarzy na stronę
    page_size = min(100, max_results)

    print(f"Rozpoczynam pobieranie (cel: {max_results} komentarzy, sortowanie: {order})...")

    try:
        while len(comments_list) < max_results:
            # Ile jeszcze potrzebujemy na tej stronie?
            remaining = max_results - len(comments_list)
            current_page_size = min(100, remaining)

            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=current_page_size,
                order=order,
                pageToken=next_page_token  # None przy pierwszym żądaniu
            )
            response = request.execute()

            # Ekstrakcja danych z odpowiedzi JSON
            for item in response.get('items', []):
                snippet = item['snippet']
                comment_data = snippet['topLevelComment']['snippet']

                comments_list.append({
                    'author':       comment_data['authorDisplayName'],
                    'text':         comment_data['textDisplay'],
                    'published_at': comment_data['publishedAt'],
                    'likes':        comment_data['likeCount'],
                    'reply_count':  snippet.get('totalReplyCount', 0),
                })

            # Sprawdzamy czy jest następna strona
            next_page_token = response.get('nextPageToken')
            fetched_so_far = len(comments_list)
            print(f"  Pobrano łącznie: {fetched_so_far} komentarzy...")

            if not next_page_token:
                print("  Brak kolejnych stron — pobrano wszystkie dostępne komentarze.")
                break

            # Krótka przerwa żeby nie przekroczyć limitów API
            time.sleep(0.2)

    except HttpError as e:
        # Obsługa błędów API (np. przekroczenie dziennego limitu)
        print(f"  Błąd API YouTube (kod {e.resp.status}): {e.error_details}")
        if e.resp.status == 403:
            print("  Możliwe przyczyny: dzienny limit zapytań wyczerpany, "
                  "komentarze wyłączone dla tego wideo, lub nieprawidłowy klucz API.")
    except Exception as e:
        print(f"  Nieoczekiwany błąd podczas pobierania: {e}")

    df = pd.DataFrame(comments_list)

    if not df.empty:
        # Konwertujemy datę na typ datetime — przyda się do analizy szeregów czasowych
        df['published_at'] = pd.to_datetime(df['published_at'])
        df = df.sort_values('published_at').reset_index(drop=True)

    return df


# --- Testowanie skryptu ---
if __name__ == "__main__":
    TEST_VIDEO_ID = "dQw4w9WgXcQ"

    print(f"Pobieram komentarze dla wideo: {TEST_VIDEO_ID}...")
    df_comments = fetch_comments(TEST_VIDEO_ID, max_results=300, order="time")

    if not df_comments.empty:
        print(f"\nPomyślnie pobrano {len(df_comments)} komentarzy!")
        print(f"Zakres czasowy: {df_comments['published_at'].min()} → {df_comments['published_at'].max()}")
        print("\nPrzykładowe komentarze:")
        print(df_comments[['text', 'likes', 'published_at']].head(10).to_string())
    else:
        print("Nie udało się pobrać żadnych komentarzy.")