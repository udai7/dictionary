"""
Streamlit UI: Sanskrit -> English lookup + ML part-of-speech prediction.

Run:
    streamlit run app.py
"""
import os
import difflib

import joblib
import streamlit as st

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "model")
CONFIDENCE_THRESHOLD = 0.90   # the "more than 90%" rule from the assignment


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "pos_model.joblib"))
    lookup = joblib.load(os.path.join(MODEL_DIR, "lookup.joblib"))
    return model, lookup


def predict_pos(model, word: str):
    """Return (best_label, confidence) using the classifier's probabilities."""
    probs = model.predict_proba([word])[0]
    classes = model.classes_
    best = probs.argmax()
    return classes[best], float(probs[best]), dict(zip(classes, probs))


st.set_page_config(page_title="Sanskrit Dictionary + ML", page_icon="🕉️")
st.title("🕉️ Sanskrit → English Dictionary")
st.caption(
    "Type a romanized (IAST) Sanskrit word. The translation comes from the "
    "dictionary; the part of speech is predicted by a trained ML model and "
    "shown only when confidence exceeds 90%."
)

if not os.path.exists(os.path.join(MODEL_DIR, "pos_model.joblib")):
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

model, lookup = load_artifacts()

word = st.text_input("Sanskrit word", placeholder="e.g. dharma, gacchati, sundara").strip().lower()

if word:
    # --- Translation: dictionary lookup -------------------------------------
    st.subheader("Translation")
    if word in lookup:
        english, true_pos = lookup[word]
        st.success(f"**{word}** — {english}")
    else:
        english, true_pos = None, None
        close = difflib.get_close_matches(word, lookup.keys(), n=3, cutoff=0.7)
        st.warning("Word not found in the dictionary.")
        if close:
            st.write("Did you mean: " + ", ".join(f"`{c}`" for c in close) + "?")

    # --- Part of speech: ML prediction with confidence ----------------------
    st.subheader("Part of speech (predicted by the model)")
    label, conf, all_probs = predict_pos(model, word)

    if conf > CONFIDENCE_THRESHOLD:
        st.success(f"**{label}**  ·  confidence {conf:.1%}")
    else:
        st.info(
            f"Predicted **{label}** but confidence is only {conf:.1%} "
            f"(below the 90% threshold), so this is shown as uncertain."
        )

    st.progress(min(conf, 1.0))

    with st.expander("All class probabilities"):
        for cls, p in sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True):
            st.write(f"{cls}: {p:.1%}")

    if true_pos and true_pos != label:
        st.caption(
            f"Note: dictionary lists this word as **{true_pos}**; "
            f"the model guessed **{label}** from spelling alone."
        )

st.divider()
st.caption(
    f"Confidence threshold = {CONFIDENCE_THRESHOLD:.0%}. "
    "Translation = lookup. Part of speech = trained scikit-learn classifier "
    "over character n-grams."
)
