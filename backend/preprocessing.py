"""
preprocessing.py — wspólny moduł czyszczenia tekstu

Używany przez WSZYSTKIE modele (VADER, klasyfikatory ML, Transformery)
jako pierwszy, ustandaryzowany etap analizy.

Kolejność kroków:
    1. Dekodowanie HTML entities  (&#39; → ')
    2. Usunięcie tagów HTML       (<br>, <b>, itp.)
    3. Zamiana emoji na tekst     (🔥 → " ogień ")
    4. Usunięcie URLi
    5. Normalizacja białych znaków
    6. Wykrywanie języka
    7. Filtrowanie spamu / pustych komentarzy
"""

import re
import html
import pandas as pd
from langdetect import detect, LangDetectException
import emoji

# ---------------------------------------------------------------------------
# Stałe — wzorce regex
# ---------------------------------------------------------------------------

# Dopasowuje URL-e (http/https/ftp oraz skróty typu www.)
URL_PATTERN = re.compile(
    r'(https?://\S+|www\.\S+|ftp://\S+)',
    flags=re.IGNORECASE
)

# Tagi HTML (np. <br>, </b>, <a href="...">)
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

# Wielokrotne spacje / taby / nowe linie → pojedyncza spacja
WHITESPACE_PATTERN = re.compile(r'\s+')

# Znaki specjalne pozostałe po czyszczeniu (opcjonalne, delikatne)
LEFTOVER_PATTERN = re.compile(r'[^\w\s\'\"\!\?\.\,\:\;\-\(\)\#\@\&\+\=\/]')


# ---------------------------------------------------------------------------
# Funkcje pomocnicze
# ---------------------------------------------------------------------------

def decode_html(text: str) -> str:
    """Zamienia HTML entities na właściwe znaki (&#39; → ', &amp; → &)."""
    return html.unescape(text)


def remove_html_tags(text: str) -> str:
    """Usuwa tagi HTML pozostałe w tekście komentarza."""
    return HTML_TAG_PATTERN.sub(' ', text)


def convert_emoji(text: str) -> str:
    """
    Zamienia emoji na opis tekstowy (🔥 → ':fire:') jeśli biblioteka
    'emoji' jest dostępna. W przeciwnym razie usuwa emoji.
    """
    return emoji.demojize(text, delimiters=(' ', ' '))

def remove_urls(text: str) -> str:
    """Usuwa linki URL z tekstu."""
    return URL_PATTERN.sub(' ', text)


def normalize_whitespace(text: str) -> str:
    """Zastępuje wielokrotne białe znaki pojedynczą spacją i przycina."""
    return WHITESPACE_PATTERN.sub(' ', text).strip()


def detect_language(text: str) -> str:
    """
    Wykrywa język tekstu. Zwraca kod ISO 639-1 (np. 'pl', 'en', 'de').
    Jeśli wykrycie jest niemożliwe, zwraca 'unknown'.
    """
    if len(text.strip()) < 10:
        # Za krótki tekst — langdetect jest zawodny
        return 'unknown'
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'


def is_spam(text: str) -> bool:
    """
    Prosta heurystyka filtrowania spamu.
    Zwraca True jeśli komentarz wygląda jak spam.

    Kryteria:
    - Pusty lub składa się z samych spacji
    - Zawiera więcej niż 3 URLe (bot reklamowy)
    - Składa się z ponad 80% cyfr
    - Powtarzający się znak (np. "aaaaaaaaaa")
    """
    stripped = text.strip()

    if not stripped:
        return True

    # Zbyt wiele linków
    if len(URL_PATTERN.findall(stripped)) > 3:
        return True

    # Ponad 80% cyfr
    digits = sum(c.isdigit() for c in stripped)
    if len(stripped) > 5 and digits / len(stripped) > 0.8:
        return True

    # Powtarzający się znak (np. "aaaaaaaa" lub "hahahahaha" - opcjonalne)
    if re.match(r'^(.)\1{9,}$', stripped):
        return True

    return False


# ---------------------------------------------------------------------------
# Główna funkcja — czyszczenie pojedynczego tekstu
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Przetwarza surowy tekst komentarza przez pełen pipeline czyszczenia.

    Kolejność ma znaczenie — np. HTML entities trzeba dekodować
    przed usunięciem tagów HTML.
    """
    text = decode_html(text)
    text = remove_html_tags(text)
    text = convert_emoji(text)
    text = remove_urls(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Funkcja do przetwarzania całego DataFrame
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame,
                         filter_spam: bool = True,
                         detect_lang: bool = True,
                         min_length: int = 3) -> pd.DataFrame:
    """
    Przyjmuje DataFrame z komentarzami (musi mieć kolumnę 'text')
    i zwraca przetworzony DataFrame z dodatkowymi kolumnami.

    Nowe kolumny:
        text_clean  - oczyszczony tekst (do użycia przez modele AI)
        language    - wykryty język ('pl', 'en', 'unknown', ...)
        is_spam     - czy komentarz wygląda jak spam (True/False)

    Parametry:
        filter_spam - jeśli True, usuwa komentarze oznaczone jako spam
        detect_lang - jeśli True, wykrywa język (wolniejsze)
        min_length  - minimalna długość oczyszczonego tekstu (znaki)
    """
    df_out = df.copy()

    print(f"  Preprocessing: {len(df_out)} komentarzy na wejściu...")

    # Krok 1: Czyszczenie tekstu
    df_out['text_clean'] = df_out['text'].apply(clean_text)

    # Krok 2: Oznaczenie spamu
    df_out['is_spam'] = df_out['text_clean'].apply(is_spam)

    # Krok 3: Filtrowanie — usuwamy spam i zbyt krótkie teksty
    before = len(df_out)
    if filter_spam:
        df_out = df_out[~df_out['is_spam']].copy()
    df_out = df_out[df_out['text_clean'].str.len() >= min_length].copy()
    removed = before - len(df_out)
    print(f"  Usunięto {removed} komentarzy (spam / zbyt krótkie).")

    # Krok 4: Wykrywanie języka
    if detect_lang:
        print("  Wykrywanie języka... (może chwilę potrwać)")
        df_out['language'] = df_out['text_clean'].apply(detect_language)
        lang_counts = df_out['language'].value_counts().head(5).to_dict()
        print(f"  Rozkład języków (top 5): {lang_counts}")
    else:
        df_out['language'] = 'unknown'

    df_out = df_out.reset_index(drop=True)
    print(f"  Preprocessing zakończony: {len(df_out)} komentarzy na wyjściu.")
    return df_out


# ---------------------------------------------------------------------------
# Testowanie modułu
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Przykładowe dane symulujące surowe komentarze z YouTube
    test_data = {
        'text': [
            "This is AMAZING!! 🔥🔥🔥 best video ever",
            "check out my channel http://spam.com subscribe now!!!",
            "To jest świetne wideo, bardzo mi się podoba 😊",
            "&#39;Never gonna give you up&#39; — klasyk na wieki! <br>",
            "aaaaaaaaaaaaaaaaaaaaaaaaaa",
            "lol",
            "I can&#39;t believe this song is still so good after all these years...",
            "Pierwszy raz słyszę i już zakochana w tej piosence ❤️",
            "12345678901234567890",
            "Great song! https://www.youtube.com/watch?v=fake just amazing.",
        ],
        'likes': [100, 0, 45, 200, 1, 5, 88, 30, 0, 12],
        'published_at': pd.date_range('2023-01-01', periods=10, freq='D'),
    }
    df_raw = pd.DataFrame(test_data)

    print("=== Test modułu preprocessing.py ===\n")
    print("Dane wejściowe:")
    print(df_raw[['text', 'likes']].to_string())
    print()

    df_clean = preprocess_dataframe(df_raw, filter_spam=True, detect_lang=True)

    print("\nWynik po preprocessingu:")
    print(df_clean[['text', 'text_clean', 'language', 'is_spam']].to_string())
