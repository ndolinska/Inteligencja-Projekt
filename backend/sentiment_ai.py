"""
sentiment_ai.py — analiza sentymentu i emocji komentarzy YouTube

Moduł zawiera dwa niezależne analizatory:

1. VADER (Valence Aware Dictionary and sEntiment Reasoner)
   - Klasyfikator słownikowy zoptymalizowany pod media społecznościowe
   - Rozpoznaje wielkie litery, wykrzykniki, emotikony, slang
   - Działa bez GPU, bardzo szybki (miliony komentarzy/minutę)
   - Wynik: Pozytywny / Neutralny / Negatywny + compound score [-1, 1]

2. GoEmotions (SamLowe/roberta-base-go_emotions)
   - Model RoBERTa wytrenowany przez Google na 58k komentarzy z Reddita
   - 28 szczegółowych kategorii emocji (radość, gniew, strach, smutek, ...)
   - Wykracza poza materiał wykładu — bezpośrednio realizuje
     "dynamikę emocji" z tytułu projektu
   - Źródło: Demszky et al. (2020). "GoEmotions: A Dataset of Fine-Grained
     Emotions." arXiv:2005.00547

Relacja między modułami:
    VADER → klasyfikacja sentymentu (pos/neg/neu) — szybka, do szeregów czasowych
    GoEmotions → głęboka analiza emocji — do wykresów radarowych i topic modeling
"""

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline as hf_pipeline

# Pobieramy słownik VADER (uruchomi się tylko raz)
nltk.download('vader_lexicon', quiet=True)

# Inicjalizacja analizatora VADER (singleton — ładujemy raz)
_sia = SentimentIntensityAnalyzer()

# GoEmotions — ładowany leniwie przy pierwszym użyciu
_emotion_pipe = None


# ---------------------------------------------------------------------------
# 28 etykiet modelu GoEmotions z polskimi tłumaczeniami
# ---------------------------------------------------------------------------

EMOTION_LABELS_PL = {
    'admiration':    'podziw',
    'amusement':     'rozbawienie',
    'anger':         'gniew',
    'annoyance':     'irytacja',
    'approval':      'aprobata',
    'caring':        'troska',
    'confusion':     'dezorientacja',
    'curiosity':     'ciekawość',
    'desire':        'pragnienie',
    'disappointment':'rozczarowanie',
    'disapproval':   'dezaprobata',
    'disgust':       'wstręt',
    'embarrassment': 'zażenowanie',
    'excitement':    'ekscytacja',
    'fear':          'strach',
    'gratitude':     'wdzięczność',
    'grief':         'żal',
    'joy':           'radość',
    'love':          'miłość',
    'nervousness':   'nerwowość',
    'optimism':      'optymizm',
    'pride':         'duma',
    'realization':   'olśnienie',
    'relief':        'ulga',
    'remorse':       'wyrzuty sumienia',
    'sadness':       'smutek',
    'surprise':      'zaskoczenie',
    'neutral':       'neutralny',
}

# Grupowanie 28 emocji w 5 makro-kategorii (do uproszczonych wizualizacji)
EMOTION_GROUPS = {
    'Pozytywne':    {'admiration', 'amusement', 'approval', 'caring',
                     'excitement', 'gratitude', 'joy', 'love',
                     'optimism', 'pride', 'relief'},
    'Negatywne':    {'anger', 'annoyance', 'disappointment', 'disapproval',
                     'disgust', 'embarrassment', 'fear', 'grief',
                     'nervousness', 'remorse', 'sadness'},
    'Zaskoczenie':  {'surprise', 'realization'},
    'Ambiwalentne': {'confusion', 'curiosity', 'desire'},
    'Neutralne':    {'neutral'},
}


def _emotion_to_group(emotion: str) -> str:
    """Zwraca makro-kategorię dla danej etykiety emocji."""
    for group, emotions in EMOTION_GROUPS.items():
        if emotion in emotions:
            return group
    return 'Neutralne'

# VADER — analiza sentymentu

def analyze_sentiment_vader(text: str) -> dict:
    """
    Analizuje sentyment pojedynczego tekstu za pomocą VADER.

    Zwraca słownik:
        label  : 'Pozytywny' / 'Neutralny' / 'Negatywny'
        score  : compound score od -1.0 (max negatywny) do 1.0 (max pozytywny)
        details: surowe wyniki VADER (neg, neu, pos, compound)
    """
    scores  = _sia.polarity_scores(text)
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

# GoEmotions — głęboka analiza 28 kategorii emocji

def _load_emotion_model():
    """Ładuje model GoEmotions przy pierwszym użyciu (lazy loading)."""
    global _emotion_pipe
    if _emotion_pipe is None:
        print("  Ładowanie modelu GoEmotions (SamLowe/roberta-base-go_emotions)...")
        print("  (Pierwsze uruchomienie: pobieranie ~500 MB, może potrwać chwilę.)")
        _emotion_pipe = hf_pipeline(
            task="text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,       # zwracamy wszystkie 28 emocji z prawdopodobieństwami
            truncation=True,
            max_length=512,
            device=-1,        # CPU; zmień na 0 jeśli masz GPU
        )
        print("  Model GoEmotions załadowany.")
    return _emotion_pipe


def analyze_emotions_single(text: str, top_n: int = 3) -> dict:
    """
    Analizuje emocje w pojedynczym tekście za pomocą GoEmotions.

    Zwraca słownik:
        top_emotion     : dominująca emocja (angielska etykieta)
        top_emotion_pl  : dominująca emocja (po polsku)
        emotion_group   : makro-kategoria ('Pozytywne', 'Negatywne', ...)
        top_n_emotions  : lista N najsilniejszych emocji z wynikami
        all_scores      : pełny słownik {emocja: prawdopodobieństwo}
    """
    pipe = _load_emotion_model()
    raw  = pipe([text])[0]   # lista słowników [{label, score}, ...]

    # Sortujemy malejąco po wyniku
    sorted_emotions = sorted(raw, key=lambda x: x['score'], reverse=True)
    top             = sorted_emotions[0]

    return {
        'top_emotion':    top['label'],
        'top_emotion_pl': EMOTION_LABELS_PL.get(top['label'], top['label']),
        'emotion_group':  _emotion_to_group(top['label']),
        'top_n_emotions': [
            {
                'label':    e['label'],
                'label_pl': EMOTION_LABELS_PL.get(e['label'], e['label']),
                'score':    round(e['score'], 4),
            }
            for e in sorted_emotions[:top_n]
        ],
        'all_scores': {e['label']: round(e['score'], 4) for e in sorted_emotions},
    }


def analyze_emotions_dataframe(df: pd.DataFrame,
                                text_col: str = 'text_clean',
                                batch_size: int = 32) -> pd.DataFrame:
    """
    Analizuje emocje dla całego DataFrame w batchach.

    Dodaje kolumny:
        emotion          : dominująca emocja (EN)
        emotion_pl       : dominująca emocja (PL)
        emotion_group    : makro-kategoria emocji
        emotion_score    : pewność dominującej emocji [0, 1]
    """
    pipe   = _load_emotion_model()
    texts  = df[text_col].tolist()
    total  = len(texts)

    top_emotions   = []
    top_emotions_pl= []
    emotion_groups = []
    emotion_scores = []

    print(f"  Analiza emocji: {total} komentarzy (batch_size={batch_size})...")

    for i in range(0, total, batch_size):
        batch   = texts[i : i + batch_size]
        results = pipe(batch)

        for result in results:
            sorted_r = sorted(result, key=lambda x: x['score'], reverse=True)
            top = sorted_r[0]
            top_emotions.append(top['label'])
            top_emotions_pl.append(EMOTION_LABELS_PL.get(top['label'], top['label']))
            emotion_groups.append(_emotion_to_group(top['label']))
            emotion_scores.append(round(top['score'], 4))

        print(f"  Postęp: {min(i + batch_size, total)}/{total}", end='\r')

    print()

    df_out = df.copy()
    df_out['emotion']       = top_emotions
    df_out['emotion_pl']    = top_emotions_pl
    df_out['emotion_group'] = emotion_groups
    df_out['emotion_score'] = emotion_scores
    return df_out


# Testowanie modułu

if __name__ == "__main__":
    print("=== Test modułu sentiment_ai.py ===\n")

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

    # --- VADER ---
    print("--- VADER (sentyment) ---\n")
    print(f"{'Komentarz':52s} | {'Etykieta':12s} | Score")
    print("-" * 75)
    for comment in test_comments:
        result = analyze_sentiment_vader(comment)
        short  = comment[:49] + "..." if len(comment) > 52 else comment
        print(f"{short:52s} | {result['label']:12s} | {result['score']:+.3f}")

    # --- GoEmotions ---
    print("\n--- GoEmotions (28 emocji) ---\n")
    print(f"{'Komentarz':45s} | {'Emocja (PL)':18s} | {'Grupa':12s} | Top-3")
    print("-" * 110)
    for comment in test_comments:
        result  = analyze_emotions_single(comment, top_n=3)
        short   = comment[:42] + "..." if len(comment) > 45 else comment
        top3    = ", ".join(
            f"{e['label_pl']} ({e['score']:.0%})"
            for e in result['top_n_emotions']
        )
        print(f"{short:45s} | {result['top_emotion_pl']:18s} | "
              f"{result['emotion_group']:12s} | {top3}")

    # --- DataFrame ---
    print("\n--- Test na DataFrame ---")
    df_test = pd.DataFrame({'text_clean': test_comments})
    df_vader = analyze_dataframe(df_test, text_col='text_clean')
    print("\nVADER na DataFrame:")
    print(df_vader[['text_clean', 'sentiment', 'sentiment_score']].to_string())

    df_emotions = analyze_emotions_dataframe(df_test, text_col='text_clean')
    print("\nGoEmotions na DataFrame:")
    print(df_emotions[['text_clean', 'emotion_pl', 'emotion_group', 'emotion_score']].to_string())