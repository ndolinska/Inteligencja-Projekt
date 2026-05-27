"""
transformer_model.py — analiza sentymentu modelem Transformer

Model: cardiffnlp/twitter-xlm-roberta-base-sentiment
    - Architektura: XLM-RoBERTa (Cross-lingual Language Model)
    - Pre-trening: 100 języków, w tym polski i angielski
    - Fine-tuning: ~198 mln tweetów z Twittera (Cardiff NLP, 2022)
    - Klasy: negative / neutral / positive
    - Rozmiar: ~1.1 GB (pobierany przy pierwszym uruchomieniu)

Dlaczego ten model (uzasadnienie do raportu):
    W odróżnieniu od VADER i TF-IDF, XLM-RoBERTa rozumie kontekst
    i kolejność słów dzięki mechanizmowi self-attention. Przykład:
        "not bad"  → VADER: negatywny (widzi "bad")
                   → XLM-RoBERTa: pozytywny (rozumie negację)
    Dodatkowo jest wielojęzyczny — ten sam model analizuje komentarze
    po polsku i angielsku bez osobnego treningu.

Źródło modelu:
    Barbieri et al. (2022). "XLM-T: Multilingual Language Models in Twitter
    for Sentiment Analysis and Beyond." arXiv:2104.12250
    https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment

Interfejs identyczny jak SentimentClassifier w ml_classifiers.py,
co umożliwia bezpośrednie porównanie wyników.
"""

import time
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import warnings
warnings.filterwarnings('ignore')

# Nazwa modelu na HuggingFace Hub
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Mapowanie etykiet modelu na polskie nazwy
# Model zwraca: 'negative', 'neutral', 'positive'
LABEL_MAP = {
    'negative': 'Negatywny',
    'neutral':  'Neutralny',
    'positive': 'Pozytywny',
}

# Mapowanie etykiet liczbowych (tweet_eval) na nazwy modelu
# tweet_eval: 0=negative, 1=neutral, 2=positive
TWEET_EVAL_TO_MODEL = {0: 'negative', 1: 'neutral', 2: 'positive'}


# ---------------------------------------------------------------------------
# Klasa modelu
# ---------------------------------------------------------------------------

class TransformerSentimentModel:
    """
    Klasyfikator sentymentu oparty na XLM-RoBERTa.

    Używa pre-trenowanego modelu twitter-xlm-roberta-base-sentiment,
    który nie wymaga dodatkowego treningu — działa od razu po załadowaniu.

    Parametry:
        model_name  : nazwa modelu z HuggingFace Hub
        batch_size  : liczba tekstów przetwarzanych jednocześnie
                      (większy = szybciej, ale wymaga więcej RAM)
        max_length  : maksymalna długość tekstu w tokenach (limit modelu: 512)
    """

    def __init__(self,
                 model_name: str = MODEL_NAME,
                 batch_size: int = 32,
                 max_length: int = 128):

        self.model_name  = model_name
        self.batch_size  = batch_size
        self.max_length  = max_length
        self._pipe       = None   # pipeline ładowany leniwie (przy pierwszym użyciu)
        self.model_label = "XLM-RoBERTa"

    def _load(self):
        """
        Ładuje model i tokenizer z HuggingFace Hub.
        Wywoływane automatycznie przy pierwszym użyciu predict_*.
        Pierwsze wywołanie pobiera model (~1.1 GB) — kolejne używają cache.
        """
        if self._pipe is not None:
            return  # już załadowany

        print(f"  Ładowanie modelu {self.model_name}...")
        print("  (Przy pierwszym uruchomieniu pobieranie może potrwać kilka minut.)")
        t0 = time.time()

        self._pipe = pipeline(
            task="text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            max_length=self.max_length,
            truncation=True,        # komentarze dłuższe niż max_length są przycinane
            padding=True,
            top_k=None,             # zwracamy prawdopodobieństwa dla wszystkich klas
            device=-1,              # CPU; zmień na 0 jeśli masz GPU z CUDA
        )

        elapsed = time.time() - t0
        print(f"  Model załadowany ({elapsed:.1f}s).")

    # ------------------------------------------------------------------
    # Predykcja
    # ------------------------------------------------------------------

    def predict_single(self, text: str) -> str:
        """
        Klasyfikuje pojedynczy tekst.
        Zwraca: 'Pozytywny', 'Neutralny' lub 'Negatywny'
        """
        self._load()
        # pipeline zwraca listę list wyników (top_k=None)
        results = self._pipe([text])[0]
        best = max(results, key=lambda x: x['score'])
        return LABEL_MAP[best['label']]

    def predict_proba_single(self, text: str) -> dict:
        """
        Zwraca prawdopodobieństwa dla każdej klasy.
        Przydatne do wizualizacji w dashboardzie.
        """
        self._load()
        results = self._pipe([text])[0]
        return {
            LABEL_MAP[r['label']]: round(r['score'], 4)
            for r in sorted(results, key=lambda x: x['label'])
        }

    def predict_dataframe(self, df: pd.DataFrame,
                          text_col: str = 'text_clean') -> pd.DataFrame:
        """
        Klasyfikuje cały DataFrame w batchach.
        Dodaje kolumny: 'sentiment_transformer' i 'sentiment_transformer_score'.
        """
        self._load()
        df_out   = df.copy()
        texts    = df_out[text_col].tolist()
        labels   = []
        scores   = []

        total    = len(texts)
        print(f"  Klasyfikacja {total} komentarzy (batch_size={self.batch_size})...")

        for i in range(0, total, self.batch_size):
            batch   = texts[i : i + self.batch_size]
            results = self._pipe(batch)
            for result in results:
                best = max(result, key=lambda x: x['score'])
                labels.append(LABEL_MAP[best['label']])
                scores.append(round(best['score'], 4))

            done = min(i + self.batch_size, total)
            print(f"  Przetworzono: {done}/{total}", end='\r')

        print()  # nowa linia po \r
        df_out['sentiment_transformer']       = labels
        df_out['sentiment_transformer_score'] = scores
        return df_out

    # ------------------------------------------------------------------
    # Ewaluacja — do porównania z ml_classifiers.py
    # ------------------------------------------------------------------

    def evaluate(self, X_test: list, y_test: list) -> dict:
        """
        Ewaluuje model na zbiorze testowym i wyświetla metryki.
        y_test powinien zawierać etykiety liczbowe (0/1/2) jak w tweet_eval.
        Zwraca słownik wyników zgodny z formatem ml_classifiers.py.
        """
        self._load()
        print(f"\n  Ewaluacja na {len(X_test)} próbkach...")
        print("  (Transformer jest wolniejszy — poczekaj chwilę.)")

        # Predykcja całego zbioru testowego w batchach
        pred_labels_pl = []
        for i in range(0, len(X_test), self.batch_size):
            batch   = X_test[i : i + self.batch_size]
            results = self._pipe(batch)
            for result in results:
                best = max(result, key=lambda x: x['score'])
                pred_labels_pl.append(LABEL_MAP[best['label']])
            print(f"  Postęp: {min(i+self.batch_size, len(X_test))}/{len(X_test)}", end='\r')
        print()

        # Konwertujemy etykiety numeryczne tweet_eval na polskie nazwy
        label_map_pl = {0: 'Negatywny', 1: 'Neutralny', 2: 'Pozytywny'}
        true_labels_pl = [label_map_pl[y] for y in y_test]

        all_labels = ['Negatywny', 'Neutralny', 'Pozytywny']
        acc    = accuracy_score(true_labels_pl, pred_labels_pl)
        f1_mac = f1_score(true_labels_pl, pred_labels_pl, average='macro',    labels=all_labels)
        f1_wei = f1_score(true_labels_pl, pred_labels_pl, average='weighted', labels=all_labels)
        cm     = confusion_matrix(true_labels_pl, pred_labels_pl, labels=all_labels)

        print(f"\n{'='*55}")
        print(f"  WYNIKI: {self.model_label}")
        print(f"{'='*55}")
        print(f"  Accuracy:          {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  F1 (macro avg):    {f1_mac:.4f}")
        print(f"  F1 (weighted avg): {f1_wei:.4f}")
        print(f"\n  Raport per klasa:")
        print(classification_report(
            true_labels_pl, pred_labels_pl,
            labels=all_labels, digits=4,
        ))
        print(f"  Macierz pomyłek:")
        cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
        print(cm_df.to_string())
        print(f"{'='*55}\n")

        return {
            'model':       self.model_label,
            'accuracy':    round(acc, 4),
            'f1_macro':    round(f1_mac, 4),
            'f1_weighted': round(f1_wei, 4),
        }


# ---------------------------------------------------------------------------
# Porównanie WSZYSTKICH modeli — główna funkcja badawcza
# ---------------------------------------------------------------------------

def full_model_comparison(n_test_samples: int = 1000,
                          save_csv: bool = True) -> pd.DataFrame:
    """
    Trenuje / ładuje wszystkie cztery modele i porównuje je na tym samym
    zbiorze testowym z tweet_eval. Wynik to tabela gotowa do raportu.

    Kolejność modeli (od najprostszego do najbardziej zaawansowanego):
        1. TF-IDF + Naive Bayes
        2. TF-IDF + Logistic Regression
        3. VADER
        4. XLM-RoBERTa (ten plik)

    Parametry:
        n_test_samples  : liczba próbek testowych (domyślnie 1000 dla szybkości)
        save_csv        : zapis wyników do CSV
    """
    from ml_classifiers import SentimentClassifier, load_tweet_eval_dataset
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)

    print("\n" + "="*55)
    print("  PEŁNE PORÓWNANIE MODELI SENTYMENTU")
    print("  Dataset: tweet_eval/sentiment (SemEval-2017)")
    print("="*55)

    # Ładujemy dane
    X_train, y_train, X_test, y_test = load_tweet_eval_dataset()

    # Ograniczamy zbiór testowy dla szybszej ewaluacji Transformera
    if n_test_samples and n_test_samples < len(X_test):
        idx = np.random.default_rng(42).choice(len(X_test), n_test_samples, replace=False)
        X_test_sub = [X_test[i] for i in idx]
        y_test_sub = [y_test[i] for i in idx]
        print(f"  Używam losowej próbki {n_test_samples} z {len(X_test)} próbek testowych.")
    else:
        X_test_sub, y_test_sub = list(X_test), list(y_test)

    results = []

    # --- Model 1 & 2: TF-IDF klasyfikatory ---
    for model_type in ['nb', 'lr']:
        clf = SentimentClassifier(model_type=model_type)
        clf.train(X_train, y_train)
        metrics = clf.evaluate(X_test_sub, y_test_sub)
        results.append(metrics)

    # --- Model 3: VADER ---
    print("\n  Ewaluacja: VADER...")
    label_map_vader = {0: 'Negatywny', 1: 'Neutralny', 2: 'Pozytywny'}
    true_labels = [label_map_vader[y] for y in y_test_sub]
    sia = SentimentIntensityAnalyzer()

    vader_preds = []
    for text in X_test_sub:
        score = sia.polarity_scores(text)['compound']
        if score >= 0.05:
            vader_preds.append('Pozytywny')
        elif score <= -0.05:
            vader_preds.append('Negatywny')
        else:
            vader_preds.append('Neutralny')

    all_labels = ['Negatywny', 'Neutralny', 'Pozytywny']
    vader_metrics = {
        'model':       'VADER',
        'accuracy':    round(accuracy_score(true_labels, vader_preds), 4),
        'f1_macro':    round(f1_score(true_labels, vader_preds, average='macro',    labels=all_labels, zero_division=0), 4),
        'f1_weighted': round(f1_score(true_labels, vader_preds, average='weighted', labels=all_labels, zero_division=0), 4),
    }
    print(f"  VADER — Accuracy: {vader_metrics['accuracy']:.4f} | "
          f"F1 macro: {vader_metrics['f1_macro']:.4f}")
    results.append(vader_metrics)

    # --- Model 4: XLM-RoBERTa ---
    print("\n  Ewaluacja: XLM-RoBERTa...")
    transformer = TransformerSentimentModel(batch_size=64)
    transformer_metrics = transformer.evaluate(X_test_sub, y_test_sub)
    results.append(transformer_metrics)

    # --- Zestawienie końcowe ---
    df_results = pd.DataFrame(results).set_index('model')
    df_results = df_results.sort_values('f1_macro', ascending=False)

    print("\n" + "="*55)
    print("  ZESTAWIENIE KOŃCOWE (posortowane wg F1 macro)")
    print("="*55)
    print(df_results.to_string())
    print("="*55)

    if save_csv:
        path = "full_model_comparison.csv"
        df_results.to_csv(path)
        print(f"\n  Wyniki zapisane do: {path}")

    return df_results


# ---------------------------------------------------------------------------
# Testowanie modułu
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Test modułu transformer_model.py ===\n")

    model = TransformerSentimentModel()

    # --- Test 1: Predykcja na przykładowych komentarzach ---
    print("--- Test 1: Predykcja na przykładowych komentarzach ---\n")
    test_comments = [
        # Angielskie — oczywiste przypadki
        "This is absolutely the best song I have ever heard!!",
        "I hate this video, complete waste of my time.",
        "The video was uploaded in 2009. Just a fact.",
        # Negacja — problem dla VADER, nie dla Transformera
        "not bad at all, actually pretty good",
        "I can't say I dislike this, it's surprisingly nice",
        # Polskie
        "Absolutnie fenomenalne wykonanie, jestem zachwycona!",
        "Okropne, nigdy więcej nie chcę tego słuchać.",
        "Oglądałam już kilka razy, za każdym razem daje mi dreszcze ❤️",
        # Krótkie / niejednoznaczne
        "ok i guess",
        "interesting",
    ]

    print(f"{'Komentarz':52s} | {'Predykcja':12s} | Pewność")
    print("-" * 80)
    for comment in test_comments:
        label = model.predict_single(comment)
        probs = model.predict_proba_single(comment)
        best_prob = max(probs.values())
        short = comment[:49] + "..." if len(comment) > 52 else comment
        print(f"{short:52s} | {label:12s} | {best_prob:.2%}")

    # --- Test 2: Pełne porównanie wszystkich modeli ---
    print("\n--- Test 2: Pełne porównanie wszystkich modeli ---")
    print("(Uwaga: ewaluacja Transformera na 500 próbkach może potrwać 2-5 minut)")
    full_model_comparison(n_test_samples=500, save_csv=True)
