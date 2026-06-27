"""
Train the part-of-speech classifier on Sanskrit headwords.

What is actually "learned":
    The model predicts the part of speech (noun / verb / adjective / adverb /
    pronoun / indeclinable) from the *spelling* of a romanized Sanskrit word.
    This works because Sanskrit morphology is highly regular -- endings like
    -ati, -ana, -tva, -in, etc. strongly signal the grammatical category.

    We use character n-gram features (TF-IDF over 2-4 char sequences, including
    word boundaries) feeding a calibrated Logistic Regression classifier.
    `predict_proba` gives a genuine confidence score, which the Streamlit app
    thresholds at 90%.

The English *translation* is NOT learned -- true translation needs a neural
sequence model and far more data. Instead we save the dictionary as a lookup
table; the app returns the stored meaning for known words.

Outputs (in model/):
    pos_model.joblib   - trained sklearn pipeline
    lookup.joblib      - {sanskrit_word: (english, pos)} dictionary
    metrics.txt        - accuracy report on a held-out test split
"""
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "model")


def load_data() -> pd.DataFrame:
    """Prefer the full downloaded dictionary; fall back to the bundled sample."""
    full = os.path.join(HERE, "data", "dictionary.csv")
    sample = os.path.join(HERE, "data", "sample_dictionary.csv")
    path = full if os.path.exists(full) else sample
    print(f"Loading data from: {os.path.relpath(path, HERE)}")
    df = pd.read_csv(path)
    df = df.dropna(subset=["sanskrit", "english", "pos"])
    df["sanskrit"] = df["sanskrit"].str.strip().str.lower()
    df["pos"] = df["pos"].str.strip().str.lower()
    df = df.drop_duplicates(subset=["sanskrit"])
    return df


def build_model() -> Pipeline:
    return Pipeline([
        ("features", TfidfVectorizer(
            analyzer="char_wb",   # character n-grams, respecting word boundaries
            ngram_range=(2, 4),
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=4.0,
            class_weight="balanced",   # helps rarer POS classes
        )),
    ])


def main():
    df = load_data()
    print(f"{len(df)} unique words across {df['pos'].nunique()} POS classes")
    print(df["pos"].value_counts().to_string())

    X = df["sanskrit"].to_numpy(dtype=object)
    y = df["pos"].to_numpy(dtype=object)

    # Stratify so every POS appears in train and test. With a tiny sample some
    # classes may be too small to stratify -- fall back gracefully.
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)

    model = build_model()
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    report = classification_report(y_te, pred, zero_division=0)
    print(f"\nHeld-out accuracy: {acc:.1%}\n")
    print(report)

    # Retrain on ALL data for the final saved model (more data = better).
    final = build_model()
    final.fit(X, y)

    lookup = {row.sanskrit: (row.english, row.pos) for row in df.itertuples()}

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final, os.path.join(MODEL_DIR, "pos_model.joblib"))
    joblib.dump(lookup, os.path.join(MODEL_DIR, "lookup.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.txt"), "w") as f:
        f.write(f"Held-out accuracy: {acc:.4f}\n\n{report}\n")

    print(f"Saved model and lookup table to {os.path.relpath(MODEL_DIR, HERE)}/")


if __name__ == "__main__":
    main()
