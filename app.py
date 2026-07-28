"""
Streamlit app for Iris Flower Classification.
Loads a pre-trained Logistic Regression model and predicts species
based on user-provided flower measurements.
"""

import streamlit as st
import pandas as pd
import joblib

# --- Configuration ---
MODEL_PATH = "iris_model.joblib"
FEATURE_NAMES = ["sepal length (cm)", "sepal width (cm)",
                  "petal length (cm)", "petal width (cm)"]
SPECIES_MAP = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
SPECIES_EMOJI = {"Setosa": "🌸", "Versicolor": "🌺", "Virginica": "🌷"}

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

# --- Custom Styling + Floating Petals Animation ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #fdf6f0 0%, #f3e6f5 100%);
        overflow: hidden;
    }

    .petal {
        position: fixed;
        top: -5%;
        font-size: 28px;
        opacity: 0.7;
        z-index: 0;
        animation-name: fall;
        animation-timing-function: linear;
        animation-iteration-count: infinite;
        pointer-events: none;
    }

    @keyframes fall {
        0%   { transform: translateY(0) rotate(0deg); opacity: 0.7; }
        100% { transform: translateY(110vh) rotate(360deg); opacity: 0.3; }
    }

    .petal:nth-child(1) { left: 5%;  animation-duration: 14s; animation-delay: 0s; }
    .petal:nth-child(2) { left: 20%; animation-duration: 18s; animation-delay: 2s; }
    .petal:nth-child(3) { left: 35%; animation-duration: 12s; animation-delay: 4s; }
    .petal:nth-child(4) { left: 50%; animation-duration: 20s; animation-delay: 1s; }
    .petal:nth-child(5) { left: 65%; animation-duration: 15s; animation-delay: 3s; }
    .petal:nth-child(6) { left: 80%; animation-duration: 17s; animation-delay: 5s; }
    .petal:nth-child(7) { left: 92%; animation-duration: 13s; animation-delay: 2.5s; }

    div.stButton > button {
        background-color: #b76e8a;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        border: none;
        font-weight: 600;
        position: relative;
        z-index: 1;
    }
    div.stButton > button:hover {
        background-color: #9a5470;
        color: white;
    }
    .result-box {
        background-color: white;
        border-radius: 12px;
        padding: 1.5em;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-top: 1em;
        position: relative;
        z-index: 1;
    }
    .block-container {
        position: relative;
        z-index: 1;
    }
    </style>

    <div class="petal">🌸</div>
    <div class="petal">🌺</div>
    <div class="petal">🌷</div>
    <div class="petal">🌸</div>
    <div class="petal">🌺</div>
    <div class="petal">🌷</div>
    <div class="petal">🌸</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path):
    """Loads the saved model once (not on every user interaction)."""
    return joblib.load(path)


def predict_species(model, measurements):
    """Takes flower measurements and returns predicted species + confidence."""
    input_df = pd.DataFrame([measurements], columns=FEATURE_NAMES)
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = probabilities[prediction]
    return SPECIES_MAP[prediction], confidence


# --- UI ---
st.markdown("<h1 style='text-align: center; color: #6a3d5f;'>🌸 Iris Flower Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Move the sliders to enter flower measurements</p>", unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3)
with col2:
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

st.write("")
_, center_col, _ = st.columns([1, 1, 1])
with center_col:
    predict_clicked = st.button("🔍 Predict Species", use_container_width=True)

if predict_clicked:
    model = load_model(MODEL_PATH)
    measurements = [sepal_length, sepal_width, petal_length, petal_width]
    species, confidence = predict_species(model, measurements)
    emoji = SPECIES_EMOJI[species]

    st.markdown(f"""
        <div class="result-box">
            <h2 style="color:#6a3d5f;">{emoji} {species}</h2>
            <p style="color:#555;">Confidence: <b>{confidence:.1%}</b></p>
        </div>
    """, unsafe_allow_html=True)