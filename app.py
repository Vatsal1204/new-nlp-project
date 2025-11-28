import streamlit as st
import joblib

st.set_page_config(layout="centered", page_title="NLP Project — Minimal")
st.title("NLP Project — Minimal UI")

# load artifacts
tfidf = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("nlp_logreg_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")  # should be saved by retrain

# show compact mapping
labels_list = list(label_encoder.classes_)
mapping_line = "  ".join([f"**{i}** → {lab}" for i, lab in enumerate(labels_list)])
st.markdown(f"<div style='padding:10px;background:#0f1720;border-radius:8px'>{mapping_line}</div>", unsafe_allow_html=True)

# input
text = st.text_area("Input sentence:", height=140, placeholder="Type something to classify...")

if st.button("Predict"):
    if not text.strip():
        st.warning("Enter a sentence first.")
    else:
        X = tfidf.transform([text])
        pred_idx = model.predict(X)[0]                 # numeric index
        pred_label = label_encoder.inverse_transform([pred_idx])[0]  # human label
        conf = model.predict_proba(X).max() if hasattr(model, "predict_proba") else None
        st.success(f"Predicted: **{pred_label}**   Confidence: {conf:.2f}")
