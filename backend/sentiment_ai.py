"""
sentiment_ai.py — analiza sentymentu komentarzy YouTube

Moduł zawiera analizator VADER:

VADER (Valence Aware Dictionary and sEntiment Reasoner)
   - Klasyfikator słownikowy zoptymalizowany pod media społecznościowe
   - Rozpoznaje wielkie litery, wykrzykniki, emotikony, slang
   - Działa bez GPU, bardzo szybki (miliony komentarzy/minutę)
   - Wynik: Pozytywny / Neutralny / Negatywny + compound score [-1, 1]
"""

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Pobieramy słownik VADER (uruchomi się tylko raz)
nltk.download('vader_lexicon', quiet=True)

# Inicjalizacja analizatora VADER (singleton — ładujemy raz)
_sia = SentimentIntensityAnalyzer()


def analyze_sentiment_vader(text: str) -> dict:
    """
    Analizuje sentyment pojedynczego tekstu za pomocą VADER.

    Zwraca słownik:
        label  : 'Pozytywny' / 'Neutralny' / 'Negatywny'
        score  : compound score od -1.0 (max negatywny) do 1.0 (max pozytywny)
        details: surowe wyniki VADER (neg, neu, pos, compound)
    """
    scores   = _sia.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        label = 'Pozytywny'
    elif compound <= -0.05:
        label = 'Negatywny'
    else:
        label = 'Neutralny'

    return {'label': label, 'score': compound, 'details': scores}


def analyze_dataframe(df: pd.DataFrame,
                      text_col: str = 'text') -> pd.DataFrame:
    """
    Analizuje sentyment całego DataFrame za pomocą VADER.
    Dodaje kolumny: 'sentiment' i 'sentiment_score'.

    Parametry:
        df       : DataFrame z komentarzami
        text_col : nazwa kolumny z tekstem (domyślnie 'text',
                   po preprocessingu użyj 'text_clean')
    """
    df_out = df.copy()
    results = df_out[text_col].apply(analyze_sentiment_vader)
    df_out['sentiment']       = results.apply(lambda r: r['label'])
    df_out['sentiment_score'] = results.apply(lambda r: r['score'])
    return df_out


# Testowanie modułu

if __name__ == "__main__":
    print("=== Test modułu sentiment_ai.py (VADER) ===\n")

    test_comments = [
        "This is absolutely the best song ever made, I love it so much!! ❤️",
        "Never gonna give you up — klasyk na wieki!",
        "I can't believe how much I still love this song after all these years.",
        "This is terrible, I hate everything about it.",
        "not bad at all, actually kind of fun",
        "The video was uploaded in 2009. It has 1.5 billion views.",
        "I'm crying happy tears listening to this, what a masterpiece.",
        "Disappointing. Expected much better from this artist.",
    ]

    print(f"{'Komentarz':52s} | {'Etykieta':12s} | Score")
    print("-" * 75)
    for comment in test_comments:
        result = analyze_sentiment_vader(comment)
        short  = comment[:49] + "..." if len(comment) > 52 else comment
        print(f"{short:52s} | {result['label']:12s} | {result['score']:+.3f}")

    print("\n--- Test na DataFrame ---")
    df_test  = pd.DataFrame({'text_clean': test_comments})
    df_vader = analyze_dataframe(df_test, text_col='text_clean')
    print(df_vader[['text_clean', 'sentiment', 'sentiment_score']].to_string())
