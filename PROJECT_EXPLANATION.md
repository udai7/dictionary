# Sanskrit → English Dictionary with Machine Learning
### Project explanation (for presenting to my teacher)

---

## 1. What the project does

I built a tool where you type a **Sanskrit word** (in Roman/IAST letters, e.g.
`dharma`, `sundara`, `gacchati`) and it gives you back two things:

1. **The English translation** of the word.
2. **The part of speech** (noun, adjective, or indeclinable), predicted by a
   machine-learning model — but only shown when the model is **more than 90%
   confident**, exactly as the assignment asked.

It runs as a simple web app built with **Streamlit** (Python).

---

## 2. The dataset

I used the **Monier-Williams Sanskrit-English Dictionary**, the standard
scholarly Sanskrit dictionary, made available digitally by the **Cologne
Digital Sanskrit Dictionaries** project.

- **~175,000 word entries** — a real, large dataset, not a toy list.
- Instead of fragile web scraping, I downloaded the already-digitised source
  file and **parsed it with my own Python script** (`build_mw_dataset.py`).
- The script does real data-engineering work:
  - converts the headwords from **SLP1 encoding into readable IAST**
    (e.g. `kuliSAsana` → `kuliśāsana`),
  - extracts the **part-of-speech tag** from the dictionary markup,
  - cleans the English definition text,
  - and writes a tidy `dictionary.csv` with columns `sanskrit, english, pos`.

POS label counts in the data: **noun ≈ 120,000, adjective ≈ 50,000,
indeclinable ≈ 5,000.**

> If the teacher specifically wanted *web scraping*, I can explain that the
> Cologne data is the same dictionary, just already digitised — parsing a
> 49 MB source file is the same skill set and avoids unreliable scraping.

---

## 3. How the machine learning works

**The model predicts the part of speech from the spelling of the word.**

This is genuinely learnable because Sanskrit grammar is very regular —
word endings strongly signal the category (certain suffixes almost always
mean "noun", others "adjective", etc.).

Technical pipeline (all `scikit-learn`):

```
word  →  character n-gram features  →  Logistic Regression  →  POS + confidence
         (TfidfVectorizer, 2–4 char,        (classifier)
          respects word boundaries)
```

- The model is **trained on ~167,000 unique words** with an 80/20 train/test
  split.
- It outputs a **real probability** for each class (`predict_proba`), which is
  the "confidence". The app applies the **90% threshold** to decide whether to
  trust the prediction.

---

## 4. The most important point to be honest about

> **The translation is a dictionary *lookup*, not a machine-learning
> *prediction*. Only the part of speech is predicted by the trained model.**

Why I designed it this way:

- Truly *generating* an English translation for a Sanskrit word is **machine
  translation**, which needs a large neural network (seq2seq), a lot of
  parallel text, and a GPU — far beyond a class project.
- For a dictionary, looking the word up is also simply *more correct*: the
  stored meaning is exact, whereas a model would only guess.
- So I put the **real ML where it actually makes sense** — classifying the
  part of speech, where the model has to generalise from spelling and produces
  a meaningful confidence score.

This is the honest, defensible engineering choice, and I can explain the
trade-off clearly.

---

## 5. Results

- **Held-out accuracy ≈ 81%** across the 3 part-of-speech classes.
- Training takes about **45 seconds** on a normal laptop.
- Examples of the 90% confidence rule in action:

| Word | Predicted POS | Confidence | Shown? |
|------|---------------|-----------|--------|
| `dharma` | noun | 93% | ✅ above 90% |
| `ca` ("and") | indeclinable | 92% | ✅ above 90% |
| `sundara` | noun | 62% | ❌ below 90% → marked uncertain |

The full report is saved automatically in `model/metrics.txt`.

---

## 6. How to demo it

```bash
cd ~/Projects/dictionary
source .venv/bin/activate
streamlit run app.py
```

Then type a few words in the box:
- a clear one (`dharma`) → shows translation + confident POS,
- an uncertain one (`sundara`) → shows the translation but says the POS
  confidence is below 90%,
- a misspelled one → the app suggests close matches.

To rebuild everything from scratch:
```bash
python build_mw_dataset.py   # download + build the 175k-row dataset
python train.py              # train the model
streamlit run app.py         # run the app
```

---

## 7. What I learned / could extend

- **Learned:** data cleaning and encoding conversion (SLP1 → IAST), turning raw
  dictionary text into ML-ready features, using prediction *probabilities* as a
  confidence threshold, and building a UI.
- **Possible extensions:**
  - add a separate **verb** class from a verb-root source,
  - accept **Devanagari** script (देव) as well as Roman letters,
  - add **fuzzy matching** so misspelled inputs still get answers,
  - try other models (Random Forest, small neural net) and compare accuracy.

---

## 8. One-paragraph summary (if I only get 30 seconds)

> "I took the full Monier-Williams Sanskrit dictionary — about 175,000 words —
> parsed and cleaned it into a dataset, and trained a scikit-learn classifier
> that predicts a word's part of speech from its spelling, with about 81%
> accuracy. The app looks up the English meaning and shows the predicted part
> of speech only when the model is over 90% confident. I kept translation as a
> lookup on purpose, because real translation needs a neural network and far
> more data — so I put the actual machine learning where it genuinely belongs."
