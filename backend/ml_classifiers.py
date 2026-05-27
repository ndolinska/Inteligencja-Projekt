"""
ml_classifiers.py — klasyfikatory ML jako bazowy szczebel porównania

Implementuje i porównuje dwa klasyczne klasyfikatory tekstowe:
    1. TF-IDF + Naive Bayes (MultinomialNB)
    2. TF-IDF + Logistic Regression

Dataset treningowy: tweet_eval/sentiment (HuggingFace Datasets)
    - 45 615 tweetów z angielskiego Twittera
    - etykiety: 0 = negatywny, 1 = neutralny, 2 = pozytywny
    - źródło mediów społecznościowych → idealne do porównania z YouTube

Cel modułu:
    - Zapewnić "środkowy szczebel" w hierarchii modeli:
        Proste ML (ten plik) → VADER → Transformery
    - Udokumentować eksperymenty i metryki do raportu
    - Udostępnić predict() kompatybilny z całym pipeline'em projektu

Użycie:
    from ml_classifiers import SentimentClassifier, compare_models
    clf = SentimentClassifier(model_type='lr')
    clf.train()
    label = clf.predict_single("This movie is absolutely fantastic!")
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Mapowanie etykiet liczbowych na tekstowe (zgodne z resztą projektu)
LABEL_MAP = {0: 'Negatywny', 1: 'Neutralny', 2: 'Pozytywny'}
LABEL_MAP_REVERSE = {v: k for k, v in LABEL_MAP.items()}


# ---------------------------------------------------------------------------
# Ładowanie danych
# ---------------------------------------------------------------------------

def load_tweet_eval_dataset():
    """
    Pobiera dataset tweet_eval/sentiment z HuggingFace Datasets.
    Zwraca (X_train, y_train, X_test, y_test) jako listy stringów i int.

    Dataset: SemEval-2017 Task 4 — Twitter Sentiment Analysis
    Licencja: Creative Commons Attribution 4.0
    Cytowanie: Rosenthal et al., 2017
    """
    try:
        from datasets import load_dataset
        print("  Pobieranie datasetu tweet_eval/sentiment z HuggingFace...")
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")

        X_train = ds['train']['text']
        y_train = ds['train']['label']
        X_test  = ds['test']['text']
        y_test  = ds['test']['label']

        print(f"  Train: {len(X_train)} próbek | Test: {len(X_test)} próbek")
        _print_label_distribution(y_train, "train")
        return X_train, y_train, X_test, y_test

    except ImportError:
        raise ImportError(
            "Brakuje biblioteki 'datasets'. Zainstaluj: pip install datasets"
        )
    except Exception as e:
        raise RuntimeError(f"Błąd podczas pobierania datasetu: {e}")


def _print_label_distribution(labels, split_name: str):
    """Wyświetla rozkład klas w zbiorze."""
    counts = pd.Series(labels).value_counts().sort_index()
    print(f"  Rozkład klas ({split_name}):")
    for label_id, count in counts.items():
        pct = count / len(labels) * 100
        print(f"    {LABEL_MAP[label_id]:12s}: {count:6d} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Klasa klasyfikatora
# ---------------------------------------------------------------------------

class SentimentClassifier:
    """
    Klasyfikator sentymentu oparty na TF-IDF + prostym modelu ML.

    Parametry:
        model_type  : 'nb'  → Naive Bayes (MultinomialNB)
                      'lr'  → Logistic Regression
        max_features: liczba cech TF-IDF (domyślnie 50 000)
        ngram_range : zakres n-gramów, np. (1,2) = unigramy i bigramy
    """

    def __init__(self, model_type: str = 'lr',
                 max_features: int = 50_000,
                 ngram_range: tuple = (1, 2)):

        if model_type not in ('nb', 'lr'):
            raise ValueError("model_type musi być 'nb' lub 'lr'")

        self.model_type   = model_type
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.is_trained   = False

        # Konfiguracja modelu
        if model_type == 'nb':
            classifier = MultinomialNB(alpha=0.1)
            model_name = "Naive Bayes"
        else:
            classifier = LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver='lbfgs',
                random_state=42,
            )
            model_name = "Logistic Regression"

        self.model_name = model_name

        # Pipeline: TF-IDF → klasyfikator
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,       # log(tf+1) zamiast surowego tf
                min_df=2,                # ignoruj słowa < 2 wystąpień
                strip_accents='unicode',
                analyzer='word',
            )),
            ('clf', classifier),
        ])

    def train(self, X_train=None, y_train=None) -> dict:
        """
        Trenuje model. Jeśli dane nie są podane, ładuje tweet_eval automatycznie.
        Zwraca metryki na zbiorze testowym.
        """
        if X_train is None or y_train is None:
            X_train, y_train, X_test, y_test = load_tweet_eval_dataset()
            self._X_test = X_test
            self._y_test = y_test
        else:
            self._X_test = None
            self._y_test = None

        print(f"\n  Trenowanie: TF-IDF + {self.model_name}...")
        print(f"  Konfiguracja: max_features={self.max_features}, "
              f"ngram_range={self.ngram_range}")

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        print(f"  Trening zakończony.")

        # Ewaluacja na zbiorze testowym
        if self._X_test is not None:
            return self.evaluate(self._X_test, self._y_test)
        return {}

    def evaluate(self, X_test, y_test) -> dict:
        """
        Ewaluuje model i wyświetla pełny raport z metrykami.
        Zwraca słownik z wynikami (do porównania modeli).
        """
        self._assert_trained()
        y_pred = self.pipeline.predict(X_test)

        acc    = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average='macro')
        f1_wei = f1_score(y_test, y_pred, average='weighted')
        cm     = confusion_matrix(y_test, y_pred)

        print(f"\n{'='*55}")
        print(f"  WYNIKI: TF-IDF + {self.model_name}")
        print(f"{'='*55}")
        print(f"  Accuracy:          {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  F1 (macro avg):    {f1_mac:.4f}")
        print(f"  F1 (weighted avg): {f1_wei:.4f}")
        print(f"\n  Raport per klasa:")
        print(classification_report(
            y_test, y_pred,
            target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
            digits=4,
        ))
        print(f"  Macierz pomyłek:")
        labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print(cm_df.to_string())
        print(f"{'='*55}\n")

        return {
            'model':       f"TF-IDF + {self.model_name}",
            'accuracy':    round(acc, 4),
            'f1_macro':    round(f1_mac, 4),
            'f1_weighted': round(f1_wei, 4),
        }

    def predict_single(self, text: str) -> str:
        """
        Klasyfikuje pojedynczy tekst.
        Zwraca: 'Pozytywny', 'Neutralny' lub 'Negatywny'
        """
        self._assert_trained()
        label_id = self.pipeline.predict([text])[0]
        return LABEL_MAP[label_id]

    def predict_proba_single(self, text: str) -> dict:
        """
        Zwraca prawdopodobieństwa dla każdej klasy.
        Przydatne do wizualizacji w dashboardzie.
        """
        self._assert_trained()
        probs = self.pipeline.predict_proba([text])[0]
        return {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probs)}

    def predict_dataframe(self, df: pd.DataFrame,
                          text_col: str = 'text_clean') -> pd.DataFrame:
        """
        Klasyfikuje cały DataFrame. Używa kolumny text_clean z preprocessingu.
        Dodaje kolumny: 'sentiment_ml' i 'sentiment_ml_model'.
        """
        self._assert_trained()
        df_out = df.copy()
        df_out['sentiment_ml'] = self.pipeline.predict(df_out[text_col]).tolist()
        df_out['sentiment_ml'] = df_out['sentiment_ml'].map(LABEL_MAP)
        df_out['sentiment_ml_model'] = self.model_name
        return df_out

    def _assert_trained(self):
        if not self.is_trained:
            raise RuntimeError(
                "Model nie jest wytrenowany. Wywołaj najpierw .train()"
            )


# ---------------------------------------------------------------------------
# Porównanie modeli
# ---------------------------------------------------------------------------

def compare_models(save_csv: bool = True) -> pd.DataFrame:
    """
    Trenuje i porównuje oba klasyfikatory (NB i LR) na tym samym datasecie.
    Zwraca DataFrame z zestawieniem wyników — gotowy do wklejenia do raportu.

    Parametry:
        save_csv - jeśli True, zapisuje wyniki do pliku CSV
    """
    print("\n" + "="*55)
    print("  PORÓWNANIE KLASYFIKATORÓW ML")
    print("  Dataset: tweet_eval/sentiment (SemEval-2017)")
    print("="*55)

    # Ładujemy dane raz i używamy dla obu modeli
    X_train, y_train, X_test, y_test = load_tweet_eval_dataset()

    results = []

    for model_type in ['nb', 'lr']:
        clf = SentimentClassifier(model_type=model_type)
        metrics = clf.train(X_train, y_train)
        metrics_eval = clf.evaluate(X_test, y_test)
        results.append(metrics_eval)

    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('model')

    print("\n  ZESTAWIENIE KOŃCOWE:")
    print(df_results.to_string())

    if save_csv:
        output_path = "ml_comparison_results.csv"
        df_results.to_csv(output_path)
        print(f"\n  Wyniki zapisane do: {output_path}")

    return df_results


# ---------------------------------------------------------------------------
# Testowanie modułu
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Test modułu ml_classifiers.py ===\n")

    # --- Test 1: Trening i ewaluacja Logistic Regression ---
    print("--- Test 1: Trening modelu LR ---")
    clf_lr = SentimentClassifier(model_type='lr')
    clf_lr.train()

    # --- Test 2: Predykcja na przykładowych komentarzach ---
    print("\n--- Test 2: Predykcja na przykładowych komentarzach ---")
    test_comments = [
        "This is absolutely the best song I have ever heard in my life!!",
        "I hate this video, it's terrible and a complete waste of time.",
        "The video was uploaded in 2009 and I'm watching in 2024.",
        "Okay I guess, nothing special about it.",
        "🔥🔥🔥 fire fire fire best video ever fire",
    ]

    print(f"\n{'Komentarz':55s} | {'Predykcja':12s} | Prawdopodobieństwa")
    print("-" * 100)
    for comment in test_comments:
        label  = clf_lr.predict_single(comment)
        probs  = clf_lr.predict_proba_single(comment)
        short  = comment[:52] + "..." if len(comment) > 55 else comment
        prob_str = " | ".join(f"{k}: {v:.2f}" for k, v in probs.items())
        print(f"{short:55s} | {label:12s} | {prob_str}")

    # --- Test 3: Porównanie NB vs LR ---
    print("\n--- Test 3: Porównanie NB vs LR ---")
    compare_models(save_csv=True)
