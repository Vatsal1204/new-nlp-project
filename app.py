# app.py — NLP Classifier with emojis + UI animations (prediction unchanged)
import streamlit as st
import joblib
import numpy as np
from pathlib import Path
import time

st.set_page_config(page_title="NLP Classifier — Emojis", layout="wide")
st.title("✨ NLP Classifier ✨")

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
    raw = to_scalar(raw_pred)
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
    return str(raw)

def map_to_clean(orig_label):
    low = str(orig_label).lower()
    if "sport" in low:
        return "sports"
    if "polit" in low:
        return "politics"
    if "comp" in low or "graphic" in low or "tech" in low:
        return "tech"
    if "sci.med" in low or "med" in low:
        return "medical"
    if "athe" in low:
        return "atheism"
    for c in CLEAN_LABELS:
        if c in low:
            return c
    return orig_label

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
labelenc = safe_load(LABELENC_PATH)

missing = [p.name for p in (TFIDF_PATH, MODEL_PATH) if not p.exists()]
if missing:
    st.error(f"Missing artifact(s): {', '.join(missing)}. Put them next to app.py and reload.")
    st.stop()

# --- UI layout ---
left, right = st.columns([1, 2])

# LEFT PANEL
with left:
    st.subheader("✨ Categories")
    for lbl in CLEAN_LABELS:
        st.markdown(f"- {EMOJI_MAP.get(lbl,'')} **{lbl.upper()}**")

    st.markdown("---")
    st.subheader("📝 Quick test sentences")

    st.markdown("**SPORTS** — The star striker scored two goals and led the team to a championship victory.")
    st.markdown("**POLITICS** — The parliament passed a new bill aimed at improving national economic stability.")
    st.markdown("**TECH** — Natural Language Processing (NLP) is a branch of AI that enables computers to understand, interpret, and generate human language.")
    st.markdown("**MEDICAL** — Doctors reported that the new treatment significantly improved patient recovery rates.")
    st.markdown("**ATHEISM** — Online forums saw heated debates about religion and secular beliefs today.")

    st.markdown("---")
    show_debug = st.checkbox("Show original label & confidence")
    show_raw = st.checkbox("Show raw model output (debug)")

# RIGHT PANEL
with right:
    st.subheader("🔍 Classify text")
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    text = st.text_area("Enter text:", value=st.session_state["input_text"], height=180)
    expected = st.selectbox("I expect (optional)", ["(none)"] + [lbl.upper() for lbl in CLEAN_LABELS])

    st.markdown("----")

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:

            # nice loading animation
            with st.spinner("🔄 Analyzing..."):
                time.sleep(0.5)

                # transform
                X = tfidf.transform([text])

                # predict
                raw_pred = model.predict(X)[0]
                orig_label = decode_raw_pred(raw_pred, model, labelenc)
                clean_label = map_to_clean(orig_label)
                conf = get_confidence(model, X)
                conf_str = f"{conf:.2f}" if conf is not None else "N/A"

            # fade-in prediction using empty container trick
            pred_box = st.empty()
            with pred_box.container():
                emoji = EMOJI_MAP.get(clean_label, "")
                st.markdown(
                    f"""
                    <div style="padding:15px;border-radius:10px;background-color:#111;
                                border:1px solid #444;animation: fadein 1s;">
                        <h3 style="color:#fff;">{emoji} Predicted category: <b>{clean_label.upper()}</b></h3>
                        <p style="opacity:0.8;">Confidence: {conf_str}</p>
                    </div>

                    <style>
                    @keyframes fadein {{
                        from {{ opacity: 0; }}
                        to {{ opacity: 1; }}
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )

            # expected check
            if expected != "(none)":
                expected_clean = expected.lower()
                if expected_clean == clean_label.lower():
                    st.success("✅ Prediction matches expected category.")
                else:
                    st.error(f"❌ Prediction does NOT match expected. (expected: {expected}, got: {clean_label.upper()})")

            # debug info
            if show_debug:
                st.write("**Original model label:**", orig_label)
                st.write("**Confidence:**", conf_str)

            if show_raw:
                st.write("**Raw model output:**", to_scalar(raw_pred))

# footer
st.markdown("---")
st.caption("✨ Enhanced UI with animations — predictions and logic remain unchanged.")

