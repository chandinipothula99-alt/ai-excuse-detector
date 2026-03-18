import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- LOAD DATA ----------------
df = pd.read_csv("excuse_dataset.csv")

X = df["text"]
y = df["label"]

# ---------------- TRAIN MODEL ----------------
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# ---------------- PREDICT FUNCTION ----------------
def predict_excuse(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba) * 100

    classes = model.classes_
    prob_dict = {classes[i]: proba[i]*100 for i in range(len(classes))}

    return prediction, confidence, prob_dict, vec

# ---------------- UI DESIGN ----------------
st.set_page_config(page_title="AI Excuse Detector", page_icon="🤖", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        color: #00FFAA;
        text-align: center;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1c1f26;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI Fake Excuse Detector")
st.write("### Detect whether your excuse is real or fake!")
# Input
user_input = st.text_area("✍️ Enter your excuse:", height=100)

if st.button("🔍 Analyze Excuse"):
   if user_input:
    label, confidence, prob_dict, vec = predict_excuse(user_input)
    emoji = "✅" if label == "genuine" else "🚨"

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.subheader("🧾 Result")
    st.write(f"**Excuse:** {user_input}")

    st.write(f"### {emoji} {label.upper()}")
    st.progress(int(confidence))

    st.write(f"**Confidence:** {confidence:.1f}%")

    st.write("### 📊 Probabilities")
    st.write(f"🟢 Genuine: {prob_dict.get('genuine', 0):.1f}%")
    st.write(f"🔴 Fake: {prob_dict.get('fake', 0):.1f}%")

    if confidence < 60:
        st.warning("⚠️ Low confidence prediction")

    # Important words
    feature_names = vectorizer.get_feature_names_out()
    input_array = vec.toarray()[0]
    top_indices = input_array.argsort()[-3:]
    important_words = [feature_names[i] for i in top_indices if input_array[i] > 0]

    st.write("### 🔍 Important Words")
    st.write(important_words)

    st.markdown('</div>', unsafe_allow_html=True)