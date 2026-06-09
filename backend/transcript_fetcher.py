"""
transcript_fetcher.py
Pobiera transkrypcję wideo z YouTube (napisy auto-generowane lub ręczne).
Kompatybilny z youtube-transcript-api >= 1.0.0
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_transcript(video_id: str, preferred_langs: list[str] | None = None) -> Optional[str]:
    """
    Pobiera transkrypcję wideo YouTube.

    Strategia:
      1. api.fetch() z preferowanymi językami (szuka dowolnego typu napisu)
      2. api.list() fallback — iteruje po wszystkich dostępnych i bierze pierwszy

    Returns:
        Pełny tekst transkrypcji jako jeden string, lub None jeśli niedostępna.
    """
    if preferred_langs is None:
        preferred_langs = ['pl', 'en', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'uk', 'cs']

    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    except ImportError:
        logger.error("Brak biblioteki youtube-transcript-api — uruchom: pip install youtube-transcript-api")
        return None

    # Instancja API (wymagane w v1.x)
    api = YouTubeTranscriptApi()

    # --- Krok 1: fetch() z preferowanymi językami ---
    try:
        fetched = api.fetch(video_id, languages=preferred_langs)
        text = ' '.join(entry.text for entry in fetched if entry.text).strip()
        if text:
            logger.info("Transkrypcja pobrana (%d słów) dla %s", len(text.split()), video_id)
            return text
    except NoTranscriptFound:
        logger.info("Brak transkrypcji w preferowanych językach dla %s — próbuję fallback", video_id)
    except TranscriptsDisabled:
        logger.warning("Transkrypcje wyłączone dla %s", video_id)
        return None
    except Exception as e:
        logger.info("fetch() nie powiódł się dla %s: %s", video_id, e)

    # --- Krok 2: list() — bierzemy cokolwiek dostępnego ---
    try:
        transcript_list = api.list(video_id)
        for transcript in transcript_list:
            try:
                fetched = transcript.fetch()
                text = ' '.join(entry.text for entry in fetched if entry.text).strip()
                if text:
                    logger.info(
                        "Fallback transkrypcja: %s (%s), %d słów dla %s",
                        transcript.language_code,
                        "auto" if transcript.is_generated else "ręczna",
                        len(text.split()),
                        video_id,
                    )
                    return text
            except Exception as e:
                logger.warning("Błąd pobierania języka %s: %s", transcript.language_code, e)
                continue
    except TranscriptsDisabled:
        logger.warning("Transkrypcje wyłączone dla %s", video_id)
    except Exception as e:
        logger.warning("list() nie powiodło się dla %s: %s", video_id, e)

    logger.warning("Brak dostępnej transkrypcji dla %s", video_id)
    return None
