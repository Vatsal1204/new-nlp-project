# app.py — NLP Classifier with emojis & friendly UI
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="NLP Classifier — Emojis", layout="wide")
st.title("NLP Classifier — Clean Labels + Emojis")

# --- paths ---
TFIDF_PATH = Path("tfidf_vectorizer.joblib")
MODEL_PATH = Path("nlp_logreg_model.joblib")
LABELENC_PATH = Path("label_encoder.joblib")

# --- final cleaned labels + emoji map ---
CLEAN_LABELS = ["sports", "politics", "tech", "medical", "atheism"]
EMOJI_MAP = {
    "sports": "⚽",
    "politics": "🏛️",
    "tech": "💻",
    "medical": "🧪",
    "atheism": "🔯",
}

# --- helpers ---
def safe_load(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Warning: failed to load {path.name}: {e}")
        return None

def to_scalar(x):
    try:
        if isinstance(x, (np.generic,)):
            return x.item()
    except Exception:
        pass
    return x

def decode_raw_pred(raw_pred, model, labelenc):
    """Return original string label (best-effort) from raw model output."""
    raw = to_scalar(raw_pred)
    # try label encoder
    if labelenc is not None:
        try:
            idx = int(raw)
            return str(labelenc.inverse_transform([idx])[0])
        except Exception:
            try:
                if raw in list(labelenc.classes_):
                    return str(raw)
            except Exception:
                pass
    # try model.classes_
    if hasattr(model, "classes_"):
        try:
            idx = int(raw)
            return str(model.classes_[idx])
        except Exception:
            try:
                if raw in list(model.classes_):
                    return str(raw)
            except Exception:
                pass
    # fallback
    return str(raw)

def map_to_clean(orig_label):
    """Map many possible dataset labels into the 5 clean labels."""
    low = str(orig_label).lower()
    if "sport" in low:
        return "sports"
    if "polit" in low or "parli" in low or "gov" in low:
        return "politics"
    if "comp" in low or "graphic" in low or "tech" in low or "computer" in low:
        return "tech"
    if "sci.med" in low or "medical" in low or "med" in low or "clinic" in low:
        return "medical"
    if "athe" in low:
        return "atheism"
    for c in CLEAN_LABELS:
        if c in low:
            return c
    return orig_label  # unknown, return as-is

def get_confidence(model, X):
    try:
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X).max())
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X)
            if isinstance(df, np.ndarray):
                if df.ndim == 1:
                    return float(1/(1+np.exp(-df[0])))
                else:
                    exps = np.exp(df - np.max(df, axis=1, keepdims=True))
                    probs = exps / np.sum(exps, axis=1, keepdims=True)
                    return float(np.max(probs))
    except Exception:
        return None
    return None

# --- load artifacts safely ---
tfidf = safe_load(TFIDF_PATH)
model = safe_load(MODEL_PATH)
labelenc = safe_load(LABELENC_PATH)  # may be None

missing = [p.name for p in (TFIDF_PATH, MODEL_PATH) if not p.exists()]
if missing:
    st.error(f"Missing artifact(s): {', '.join(missing)}. Put them next to app.py and reload.")
    st.stop()

# --- UI layout ---
left, right = st.columns([1, 2])

with left:
    st.subheader("Categories (clean)")
    for lbl in CLEAN_LABELS:
        st.markdown(f"- {EMOJI_MAP.get(lbl,'')} **{lbl.upper()}**")
    st.divider()
    st.subheader("Quick test sentences")
    st.markdown("**SPORTS** — The star striker scored two goals and led the team to a championship victory.")
    st.markdown("**POLITICS** — The parliament passed a new bill aimed at improving national economic stability.")
    st.markdown("**TECH** — Researchers announced a breakthrough in artificial intelligence that boosts processing speed.")
    st.markdown("**MEDICAL** — Doctors reported that the new treatment significantly improved patient recovery rates.")
    st.markdown("**ATHEISM** — Online forums saw heated debates about religion and secular beliefs today.")
    st.divider()
    show_debug = st.checkbox("Show original label & confidence", value=False)
    show_raw = st.checkbox("Show raw model output (debug)", value=False)

with right:
    st.subheader("Classify text")
    # use session_state for the input so quick buttons can set it
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    text = st.text_area("Enter text:", value=st.session_state["input_text"], height=180, key="main_input")
    expected = st.selectbox("I expect (optional)", ["(none)"] + [lbl.upper() for lbl in CLEAN_LABELS])

    col_a, col_b, col_c = st.columns([1,1,1])
    if col_a.button("Try SPORTS"):
        st.session_state["main_input"] = "The star striker scored two goals and led the team to a championship victory."
        st.experimental_rerun()
    if col_b.button("Try POLITICS"):
        st.session_state["main_input"] = "The parliament passed a new bill aimed at improving national economic stability."
        st.experimental_rerun()
    if col_c.button("Try TECH"):
        st.session_state["main_input"] = "Researchers announced a breakthrough in artificial intelligence that boosts processing speed."
        st.experimental_rerun()
    col_d, col_e = st.columns([1,1])
    if col_d.button("Try MEDICAL"):
        st.session_state["main_input"] = "Doctors reported that the new treatment significantly improved patient recovery rates."
        st.experimental_rerun()
    if col_e.button("Try ATHEISM"):
        st.session_state["main_input"] = "Online forums saw heated debates about religion and secular beliefs today."
        st.experimental_rerun()

    st.markdown("---")

    if st.button("Predict"):
        if not text or not text.strip():
            st.warning("Enter some text first.")
        else:
            # transform
            try:
                X = tfidf.transform([text])
            except Exception as e:
                st.error(f"TF-IDF transform error: {e}")
                st.stop()

            # predict
            try:
                raw_pred = model.predict(X)[0]
            except Exception as e:
                st.error(f"Model prediction error: {e}")
                st.stop()

            orig_label = decode_raw_pred(raw_pred, model, labelenc)
            clean_label = map_to_clean(orig_label)
            conf = get_confidence(model, X)
            conf_str = f"{conf:.2f}" if conf is not None else "N/A"

            # show with emoji and colored box
            emoji = EMOJI_MAP.get(clean_label, "")
            st.markdown(f"### {emoji} Predicted category: **{clean_label.upper()}**")
            st.caption(f"(confidence: {conf_str})")

            # expected check
            if expected != "(none)":
                expected_clean = expected.lower()
                if expected_clean == clean_label.lower():
                    st.success("✅ Prediction matches expected category.")
                else:
                    st.error(f"❌ Prediction does NOT match expected. (expected: {expected}, got: {clean_label.upper()})")

            if show_debug:
                st.write("**Original model label (best-effort):**", orig_label)
                st.write("**Confidence:**", conf_str)
            if show_raw:
                st.write("**Raw model output:**", to_scalar(raw_pred))

# footer
st.markdown("---")
st.caption("Mapping heuristics convert model labels (even numeric or dataset-specific) into the five clean categories shown above.")
