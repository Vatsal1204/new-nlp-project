# app.py
import streamlit as st
from pathlib import Path
import joblib
import numpy as np

st.set_page_config(page_title="NLP Text Classifier", layout="centered")
st.title("NLP Text Classifier — Text labels")

BASE = Path.cwd()
TFIDF_F = BASE / "tfidf_vectorizer.joblib"
MODEL_F = BASE / "nlp_logreg_model.joblib"
LABELENC_F = BASE / "label_encoder.joblib"

# Optional mapping: reduce verbose dataset labels to 5 simple categories.
# Edit this dict to match the actual labels present in your encoder / model.
SIMPLIFY_MAPPING = {
    # politics variants -> politics
    "talk.politics.misc": "politics",
    "politics": "politics",
    # medical variants -> medical
    "sci.med": "medical",
    "medical": "medical",
    # tech / comp variants -> tech
    "comp.graphics": "tech",
    "computer": "tech",
    "tech": "tech",
    # atheism variants -> atheism
    "alt.atheism": "atheism",
    "atheism": "atheism",
    # sports
    "sports": "sports",
}

FALLBACK_LABELS = ["sports", "politics", "tech", "medical", "atheism"]

def safe_load(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        return e

# Check presence
missing = [p.name for p in (TFIDF_F, MODEL_F) if not p.exists()]
if missing:
    st.error(f"Missing artifact(s): {', '.join(missing)}. Place them next to app.py.")
    st.stop()

tfidf = safe_load(TFIDF_F)
model = safe_load(MODEL_F)
labelenc = None
if LABELENC_F.exists():
    labelenc = safe_load(LABELENC_F)

# If loading raised exceptions, show and stop
for obj, name in [(tfidf, TFIDF_F.name), (model, MODEL_F.name), (labelenc, LABELENC_F.name if LABELENC_F.exists() else None)]:
    if isinstance(obj, Exception):
        st.error(f"Error loading {name}: {obj}")
        st.stop()

# Determine label source and classes
if labelenc is not None and hasattr(labelenc, "classes_"):
    original_classes = list(labelenc.classes_)
    label_source = "label_encoder"
elif hasattr(model, "classes_"):
    original_classes = list(model.classes_)
    label_source = "model.classes_"
else:
    # fallback to user-friendly set
    original_classes = FALLBACK_LABELS
    label_source = "fallback"

st.markdown(f"**Label source:** `{label_source}`")
st.markdown("**Available labels (index → label):**")
st.write({i: lab for i, lab in enumerate(original_classes)})

# UI controls
simplify = st.checkbox("Simplify to 5 categories (sports, politics, tech, medical, atheism)", value=False)
st.markdown("---")
text = st.text_area("Enter text to classify", height=140, placeholder="Type news sentence here...")

def to_scalar(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    return x

def decode_prediction(raw_pred):
    """
    raw_pred may be numeric index or label string.
    Returns label_text (string) and used_source label.
    """
    raw = to_scalar(raw_pred)

    # If we have a label encoder, try inverse_transform numeric index
    if labelenc is not None:
        try:
            idx = int(raw)
            return str(labelenc.inverse_transform([idx])[0]), "label_encoder"
        except Exception:
            # maybe raw is already a label string
            try:
                if raw in list(labelenc.classes_):
                    return str(raw), "label_encoder"
            except Exception:
                pass

    # If model.classes_ exists
    if hasattr(model, "classes_"):
        try:
            idx = int(raw)
            return str(model.classes_[idx]), "model.classes_"
        except Exception:
            if raw in list(model.classes_):
                return str(raw), "model.classes_"

    # fallback if numeric and within fallback list
    try:
        idx = int(raw)
        if 0 <= idx < len(FALLBACK_LABELS):
            return FALLBACK_LABELS[idx], "fallback"
    except Exception:
        pass

    return str(raw), "raw"

def get_confidence(X):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return float(np.max(proba))
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X)
            if isinstance(df, np.ndarray):
                if df.ndim == 1:
                    val = 1 / (1 + np.exp(-df[0]))
                    return float(val)
                else:
                    exps = np.exp(df - np.max(df, axis=1, keepdims=True))
                    probs = exps / np.sum(exps, axis=1, keepdims=True)
                    return float(np.max(probs))
    except Exception:
        return None
    return None

if st.button("Predict"):
    if not text or not text.strip():
        st.warning("Please enter text to classify.")
    else:
        try:
            X = tfidf.transform([text])
        except Exception as e:
            st.error(f"TF-IDF transform error: {e}")
            st.stop()

        try:
            raw_pred = model.predict(X)[0]
        except Exception as e:
            st.error(f"Model prediction error: {e}")
            st.stop()

        label_text, used = decode_prediction(raw_pred)

        # apply optional simplification mapping
        simple_used = "none"
        if simplify:
            simple = SIMPLIFY_MAPPING.get(label_text, None)
            if simple is None:
                # try to find substring matches (robust)
                low = label_text.lower()
                if "polit" in low:
                    simple = "politics"
                elif "sci.med" in label_text or "med" in low or "clinic" in low:
                    simple = "medical"
                elif "comp" in low or "graphics" in low or "tech" in low or "computer" in low:
                    simple = "tech"
                elif "athe" in low:
                    simple = "atheism"
                elif "sport" in low:
                    simple = "sports"
                else:
                    simple = label_text  # fallback to original if unknown
            label_display = simple
            simple_used = "mapping"
        else:
            label_display = label_text

        conf = get_confidence(X)
        conf_str = f"{conf:.2f}" if conf is not None else "N/A"

        st.success(f"Predicted category: **{label_display}**")
        st.caption(f"(original: {label_text}  •  label source: {used}  •  confidence: {conf_str})")
