import os
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Ładujemy klucz z pliku .env
load_dotenv()
API_KEY = os.getenv('key')
# Inicjalizacja klienta YouTube API
youtube = build('youtube', 'v3', developerKey=API_KEY)

def fetch_comments(video_id, max_results=20):
    """
    Pobiera najpopularniejsze komentarze dla zadanego ID filmu na YouTube.
    Zwraca je w postaci tabeli Pandas DataFrame.
    """
    comments_list = []
    
    try:
        # Żądanie do API
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            order="relevance" # Pobiera najpopularniejsze ("top comments")
        )
        response = request.execute()
        
        # Ekstrakcja danych z odpowiedzi JSON
        for item in response.get('items', []):
            comment_data = item['snippet']['topLevelComment']['snippet']
            
            # Wyciągamy tylko to, co nas interesuje do analizy
            comments_list.append({
                'author': comment_data['authorDisplayName'],
                'text': comment_data['textDisplay'],
                'published_at': comment_data['publishedAt'],
                'likes': comment_data['likeCount']
            })
            
    except Exception as e:
        print(f"Wystąpił błąd podczas pobierania: {e}")
        
    # Konwersja na ładną tabelkę DataFrame (bardzo przydatne do dalszej analizy AI)
    return pd.DataFrame(comments_list)

# --- Testowanie skryptu ---
if __name__ == "__main__":
    # Przykładowe ID filmu z YouTube (to te znaki po "v=" w linku)
    # Np. dla linku https://www.youtube.com/watch?v=dQw4w9WgXcQ to jest "dQw4w9WgXcQ"
    TEST_VIDEO_ID = "dQw4w9WgXcQ" 
    
    print(f"Pobieram komentarze dla wideo: {TEST_VIDEO_ID}...")
    df_comments = fetch_comments(TEST_VIDEO_ID)
    
    if not df_comments.empty:
        print(f"\nPomyślnie pobrano {len(df_comments)} komentarzy!\n")
        # Wyświetlamy 5 pierwszych wierszy (tylko kolumny tekst i polubienia, dla czytelności)
        print(df_comments[['text', 'likes']].head(n=20))
    else:
        print("Nie udało się pobrać żadnych komentarzy.")