# app.py (improved, robust)
import streamlit as st
import joblib
from pathlib import Path
import numpy as np

st.set_page_config(layout="centered", page_title="NLP Project — Robust")
st.title("NLP Project — Robust UI")

BASE = Path.cwd()

# artifact filenames
TFIDF_F = BASE / "tfidf_vectorizer.joblib"
MODEL_F = BASE / "nlp_logreg_model.joblib"
LABELENC_F = BASE / "label_encoder.joblib"

# helper to load safely
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return e  # return exception for reporting

# Check existence and load
missing = []
if not TFIDF_F.exists():
    missing.append(str(TFIDF_F.name))
if not MODEL_F.exists():
    missing.append(str(MODEL_F.name))

if missing:
    st.error(f"Missing artifact file(s): {', '.join(missing)}. Make sure these are in the same folder as app.py.")
    st.stop()

tfidf = safe_load(TFIDF_F)
model = safe_load(MODEL_F)
label_encoder = None
if LABELENC_F.exists():
    label_encoder = safe_load(LABELENC_F)

# If loading raised exceptions, show them
for obj, name in [(tfidf, TFIDF_F.name), (model, MODEL_F.name), (label_encoder, LABELENC_F.name if LABELENC_F.exists() else None)]:
    if isinstance(obj, Exception):
        st.error(f"Error loading {name}: {obj}")
        st.stop()

# Show mapping if possible
mapping_items = []
if label_encoder is not None and hasattr(label_encoder, "classes_"):
    classes = list(label_encoder.classes_)
    mapping_items = [f"**{i}** → {lab}" for i, lab in enumerate(classes)]
elif hasattr(model, "classes_"):
    classes = list(model.classes_)
    mapping_items = [f"**{i}** → {lab}" for i, lab in enumerate(classes)]

if mapping_items:
    mapping_line = "  ".join(mapping_items)
    st.markdown(f"<div style='padding:10px;background:#0f1720;border-radius:8px'>{mapping_line}</div>", unsafe_allow_html=True)
else:
    st.info("No label mapping available (no label_encoder.joblib and model.classes_ missing). Predictions will show raw output.")

# Input area
text = st.text_area("Input sentence:", height=140, placeholder="Type something to classify...")

def get_label(pred):
    """
    pred: output from model.predict (single element)
    Returns human-readable label and a flag whether it was mapped using encoder/classes_
    """
    # if pred is numpy array or scalar convert to python type
    try:
        # convert numpy scalar to python
        if isinstance(pred, (np.generic,)):
            pred = pred.item()
    except Exception:
        pass

    # If label_encoder is available and prediction is numeric
    if label_encoder is not None:
        try:
            # If pred is numeric (int-like), inverse_transform works
            if isinstance(pred, (int, np.integer, str)) and not isinstance(pred, str):
                return label_encoder.inverse_transform([int(pred)])[0], "label_encoder"
            # if pred is numeric in string form
            try:
                maybe_int = int(pred)
                return label_encoder.inverse_transform([maybe_int])[0], "label_encoder"
            except Exception:
                pass
            # if pred is already an encoded label recognized by encoder
            # We still attempt inverse_transform only if pred in classes_
            if pred in list(label_encoder.classes_):
                return pred, "label_encoder"
        except Exception:
            pass

    # If model.classes_ exists and pred is index-like
    if hasattr(model, "classes_"):
        try:
            # if pred is numeric index
            if isinstance(pred, (int, np.integer)):
                return model.classes_[int(pred)], "model.classes_"
            # if numeric string
            try:
                maybe_int = int(pred)
                return model.classes_[maybe_int], "model.classes_"
            except Exception:
                pass
            # if pred already a label in classes_ return as-is
            if pred in list(model.classes_):
                return pred, "model.classes_"
        except Exception:
            pass

    # fallback: return pred as-is
    return pred, "raw"

if st.button("Predict"):
    if not text.strip():
        st.warning("Enter a sentence first.")
    else:
        try:
            X = tfidf.transform([text])
        except Exception as e:
            st.error(f"Error transforming input with tfidf_vectorizer: {e}")
            st.stop()

        try:
            pred_raw = model.predict(X)[0]
        except Exception as e:
            st.error(f"Error during model.predict: {e}")
            st.stop()

        pred_label, source = get_label(pred_raw)

        # confidence
        conf = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                # ensure proba shape and get probability of chosen class
                if proba.shape[1] == len(getattr(model, "classes_", proba.shape[1])):
                    # if label mapping used, find index of pred_label in model.classes_
                    if hasattr(model, "classes_"):
                        idx = list(model.classes_).index(pred_label) if pred_label in list(model.classes_) else np.argmax(proba)
                    else:
                        idx = np.argmax(proba)
                    conf = float(proba[0, idx])
                else:
                    conf = float(np.max(proba))
            elif hasattr(model, "decision_function"):
                # map decision to probability-like via softmax (best-effort)
                df = model.decision_function(X)
                if df.ndim == 1:
                    # binary: use sigmoid-like scaling
                    conf = float(1 / (1 + np.exp(-df[0])))
                else:
                    exps = np.exp(df - np.max(df, axis=1, keepdims=True))
                    probs = exps / np.sum(exps, axis=1, keepdims=True)
                    conf = float(np.max(probs))
        except Exception:
            conf = None

        conf_text = f"{conf:.2f}" if conf is not None else "N/A"
        st.success(f"Predicted: **{pred_label}**   Confidence: {conf_text}")
        st.caption(f"(label source: {source})")
