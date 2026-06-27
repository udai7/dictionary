# Sanskrit → English Dictionary with ML Part-of-Speech Prediction

A Streamlit app that, given a romanized (IAST) Sanskrit word, returns its
**English translation** (dictionary lookup) and predicts its **part of speech**
using a trained machine-learning model, shown only when the model's confidence
exceeds **90%**.

## What is and isn't "learned"

| Output | How it's produced |
|---|---|
| English translation | **Lookup** in the dictionary table. True translation needs a neural seq2seq model and far more data, so this is a lookup, not a prediction. |
| Part of speech | **Trained ML model** — a scikit-learn classifier over character n-grams (`TfidfVectorizer(char_wb, 2–6)` → calibrated `LinearSVC`). Produces a real `predict_proba` confidence that is thresholded at 90%. |

This works because Sanskrit morphology is regular: endings strongly signal the
grammatical category (noun / adjective / indeclinable).

## Dataset

The **Monier-Williams Sanskrit-English Dictionary** (Cologne Digital Sanskrit
Dictionaries), ~**175,000 entries**. `build_mw_dataset.py` downloads the raw
source, converts headwords from SLP1 to IAST, extracts POS tags and glosses,
and writes `data/dictionary.csv`.

POS label counts: noun ≈ 120k, adjective ≈ 50k, indeclinable ≈ 5k.
(MW marks m./f./n. = noun, mfn. = adjective, ind. = indeclinable.)

A small `data/sample_dictionary.csv` (~200 words) is bundled so the pipeline
runs even offline; `train.py` automatically prefers the full dictionary.

## Setup & run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python build_mw_dataset.py   # download + build data/dictionary.csv (~175k rows)
python train.py              # train the model -> model/
streamlit run app.py         # launch the UI
```

## Files

- `build_mw_dataset.py` — download & parse Monier-Williams → `data/dictionary.csv`
- `train.py` — train POS classifier, save `model/pos_model.joblib` + `model/lookup.joblib`
- `app.py` — Streamlit UI (translation lookup + POS prediction + 90% threshold)
- `data/sample_dictionary.csv` — small offline fallback dataset
- `model/metrics.txt` — held-out accuracy report

## Current results

Held-out accuracy ≈ **84%** across 4 POS classes (noun, adjective, verb,
indeclinable) on ~173k unique words. Most confident predictions (clear
noun/adjective suffixes, indeclinables, verb roots) land well above the 90%
confidence threshold. The main residual error is the genuine noun↔adjective
ambiguity — many Sanskrit words are both, with identical spelling.

## Notes / possible extensions

- Verb **roots** are now extracted from MW (entries marked with a conjugation
  class, e.g. `gam` → "to go, move"). Inflected verb forms like `gacchati` are
  not separate headwords in MW, so they are still classified by spelling.
- Some glosses retain MW grammatical notes; you can clean them further in
  `build_mw_dataset.py:clean_english`.
- To add fuzzy matching for misspellings, the app already suggests close words
  via `difflib`.
