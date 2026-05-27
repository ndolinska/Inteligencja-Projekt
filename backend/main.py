"""
main.py — FastAPI backend dla YouTube Sentiment Dashboard

Uruchomienie:
    uvicorn main:app --reload --port 8000

Endpointy:
    GET  /api/health      — sprawdzenie działania serwera
    POST /api/analyze     — pełna analiza komentarzy wideo
"""

import re
import time
import sys
import os
from collections import Counter

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from youtube_fetcher import fetch_comments
from preprocessing import preprocess_dataframe
from sentiment_ai import analyze_dataframe


# Aplikacja

app = FastAPI(
    title="YouTube Sentiment Dashboard API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict = {}

# Modele

class AnalyzeRequest(BaseModel):
    video_url:    str
    max_comments: int = 500
    order:        str = "time"   # "time" | "relevance"
    model:        str = "vader"  # "vader" | "xlm-roberta"

# Singleton — model ładowany raz, trzymany w pamięci między requestami
_transformer_model = None

def get_transformer():
    global _transformer_model
    if _transformer_model is None:
        from transformer_model import TransformerSentimentModel
        _transformer_model = TransformerSentimentModel()
    return _transformer_model

# Helpers

STOP_WORDS = {
    # EN
    'the','and','for','are','but','not','you','all','can','had','her','was',
    'one','our','out','get','has','him','his','how','its','may','now','she',
    'two','who','did','let','put','say','too','use','way','any','ago','this',
    'that','with','they','have','from','been','than','when','what','will',
    'your','more','very','just','like','know','some','time','make','into',
    'also','about','there','still','even','back','only','well','then','first',
    'than','been','have','after','over','such','much','most','each','here',
    # PL
    'się','nie','tak','jak','ale','czy','już','też','jest','jego','jej','ich',
    'nas','was','tego','tej','ten','ta','to','ze','po','do','na','w','z','o',
    'co','który','które','która','przez','przy','za','przed','pod','nad',
}


def extract_video_id(raw: str) -> str:
    patterns = [
        r'(?:v=)([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
        r'embed/([a-zA-Z0-9_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, raw.strip())
        if m:
            return m.group(1)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', raw.strip()):
        return raw.strip()
    raise ValueError(f"Nieprawidłowy URL lub ID wideo: {raw}")


def build_time_series(df: pd.DataFrame) -> list:
    if 'published_at' not in df.columns or df.empty:
        return []
    df2 = df.copy()
    df2['period'] = df2['published_at'].dt.to_period('M').astype(str)
    grouped = (
        df2.groupby(['period', 'sentiment'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    result = []
    for _, row in grouped.iterrows():
        entry = {'period': row['period']}
        for s in ['Pozytywny', 'Neutralny', 'Negatywny']:
            entry[s] = int(row.get(s, 0))
        entry['total'] = sum(entry[s] for s in ['Pozytywny', 'Neutralny', 'Negatywny'])
        result.append(entry)
    return sorted(result, key=lambda x: x['period'])


def build_word_frequency(df: pd.DataFrame, top_n: int = 60) -> list:
    col = 'text_clean' if 'text_clean' in df.columns else 'text'
    words = []
    for text in df[col].dropna():
        for w in str(text).lower().split():
            w = w.strip('.,!?;:"\'-()[]{}|')
            if len(w) >= 3 and w not in STOP_WORDS and w.isalpha():
                words.append(w)
    return [
        {'word': w, 'count': c}
        for w, c in Counter(words).most_common(top_n)
    ]


def serialize_comments(df: pd.DataFrame, sentiment: str, n: int = 5) -> list:
    subset = df[df['sentiment'] == sentiment].nlargest(n, 'likes')
    records = []
    for _, row in subset.iterrows():
        records.append({
            'text':    str(row.get('text', '')),
            'likes':   int(row.get('likes', 0)),
            'score':   float(row.get('sentiment_score', 0)),
            'date':    str(row['published_at'])[:10] if pd.notna(row.get('published_at')) else '',
        })
    return records

# Endpointy

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    t0 = time.time()

    # Parsowanie ID
    try:
        video_id = extract_video_id(req.video_url)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    # Cache — klucz zawiera model, żeby wyniki VADER i RoBERTy były oddzielne
    cache_key = f"{video_id}|{req.max_comments}|{req.order}|{req.model}"
    if cache_key in _cache:
        return {**_cache[cache_key], "from_cache": True}

    # Pobieranie
    try:
        df_raw = fetch_comments(video_id, max_results=req.max_comments, order=req.order)
    except Exception as e:
        raise HTTPException(502, detail=f"Błąd YouTube API: {e}")

    if df_raw.empty:
        raise HTTPException(404, detail="Brak komentarzy — mogą być wyłączone dla tego wideo.")

    # Preprocessing
    df_clean = preprocess_dataframe(df_raw, filter_spam=True, detect_lang=True)
    if df_clean.empty:
        raise HTTPException(422, detail="Po filtrowaniu nie zostały żadne komentarze.")

    # Sentyment — wybór modelu przez użytkownika
    if req.model == "xlm-roberta":
        df = get_transformer().analyze_dataframe(df_clean, text_col='text_clean')
    else:
        df = analyze_dataframe(df_clean, text_col='text_clean')  # VADER (domyślny)

    # Agregacja
    total  = len(df)
    counts = df['sentiment'].value_counts().to_dict()
    dist   = {s: counts.get(s, 0) for s in ['Pozytywny', 'Neutralny', 'Negatywny']}
    pct    = {s: round(v / total * 100, 1) for s, v in dist.items()}

    lang_counts = df['language'].value_counts()
    languages   = [
        {'language': l, 'count': int(c), 'percentage': round(c / total * 100, 1)}
        for l, c in lang_counts.head(8).items()
    ]

    result = {
        'video_id':          video_id,
        'model_used':        req.model,
        'total_fetched':     len(df_raw),
        'total_analyzed':    total,
        'filtered_out':      len(df_raw) - total,
        'distribution':      dist,
        'percentages':       pct,
        'time_series':       build_time_series(df),
        'languages':         languages,
        'top_positive':      serialize_comments(df, 'Pozytywny'),
        'top_negative':      serialize_comments(df, 'Negatywny'),
        'word_frequency':    build_word_frequency(df),
        'processing_time':   round(time.time() - t0, 2),
        'from_cache':        False,
    }

    _cache[cache_key] = result
    return result
