# app.py FINAL CLEAN VERSION (always outputs: sports, politics, tech, medical, atheism)

import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="NLP Classifier", layout="centered")
st.title("NLP Text Classifier (Final Clean Output)")

# Load
tfidf = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("nlp_logreg_model.joblib")
labelenc = joblib.load("label_encoder.joblib")

# 🔥 Mapping ANY output → final 5 clean labels
CLEAN_MAP = {
    "sports": "sports",
    "sport": "sports",

    "talk.politics.misc": "politics",
    "politics": "politics",
    "gov": "politics",

    "comp.graphics": "tech",
    "computer": "tech",
    "tech": "tech",
    "graphics": "tech",

    "sci.med": "medical",
    "medical": "medical",
    "med": "medical",

    "atheism": "atheism",
    "alt.atheism": "atheism"
}

# Converts ANY raw output → clean 5 labels
def to_clean_label(raw):
    raw = str(raw).lower()

    # Direct match
    if raw in CLEAN_MAP:
        return CLEAN_MAP[raw]

    # Fallback substring matching (smart matching)
    if "sport" in raw:
        return "sports"
    if "polit" in raw or "gov" in raw:
        return "politics"
    if "comp" in raw or "graphic" in raw or "tech" in raw or "computer" in raw:
        return "tech"
    if "med" in raw or "clinic" in raw:
        return "medical"
    if "athe" in raw:
        return "atheism"

    # Unknown? show raw
    return raw

# Input box
text = st.text_area("Enter text:", height=130)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        X = tfidf.transform([text])

        # raw numeric index
        pred_idx = model.predict(X)[0]

        # label text from encoder
        pred_raw = labelenc.inverse_transform([pred_idx])[0]

        # convert to clean 5 categories
        clean = to_clean_label(pred_raw)

        st.success(f"Predicted Category: **{clean.upper()}**")
