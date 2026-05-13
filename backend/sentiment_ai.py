import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Pobieramy słownik VADER (uruchomi się tylko raz)
nltk.download('vader_lexicon', quiet=True)

# Inicjalizacja analizatora
sia = SentimentIntensityAnalyzer()

def analyze_dataframe(df):
    """
    Przyjmuje DataFrame z komentarzami i dodaje kolumny z analizą sentymentu.
    """
    # Kopiujemy tabelę, żeby nie modyfikować oryginału
    df_analyzed = df.copy()
    
    sentiments = []
    scores = []
    
    for text in df_analyzed['text']:
        # VADER zwraca słownik z wynikami: neg, neu, pos oraz compound (zsumowany wynik od -1 do 1)
        score = sia.polarity_scores(text)
        compound = score['compound']
        scores.append(compound)
        
        # Prosta klasyfikacja na podstawie wyniku 'compound'
        if compound >= 0.05:
            sentiments.append('Pozytywny')
        elif compound <= -0.05:
            sentiments.append('Negatywny')
        else:
            sentiments.append('Neutralny')
            
    # Dodajemy nowe kolumny do naszej tabeli
    df_analyzed['sentiment'] = sentiments
    df_analyzed['sentiment_score'] = scores
    
    return df_analyzed

# --- Testowanie skryptu ---
if __name__ == "__main__":
    # Importujemy funkcję z naszego poprzedniego pliku!
    from youtube_fetcher import fetch_comments
    
    TEST_VIDEO_ID = "dQw4w9WgXcQ" # Rickroll (komentarze po angielsku)
    print("Pobieranie komentarzy...")
    df = fetch_comments(TEST_VIDEO_ID, max_results=20)
    
    if not df.empty:
        print("Analizowanie emocji...")
        df_results = analyze_dataframe(df)
        
        # Wyświetlamy tekst, przypisaną emocję i dokładny wynik liczbowy
        print("\nWyniki analizy:")
        print(df_results[['text', 'sentiment', 'sentiment_score']].head(10))
        
        # Małe podsumowanie dla Twojego przyszłego dashboardu
        print("\nPodsumowanie emocji:")
        print(df_results['sentiment'].value_counts())