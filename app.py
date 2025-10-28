import streamlit as st
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import uvicorn
import threading
from model import model  # 🧠 Import your Keras model from model.py

# ============================
# 🧠 BACKEND - FASTAPI APP
# ============================
app = FastAPI()

# Allow Streamlit to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define your class labels (change based on your dataset)
labels = ["Persian Cat", "Bulldog", "Siamese Cat", "Golden Retriever"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and prepare image
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    image = image.resize((224, 224))  # Adjust size to your model input
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    # Predict using the imported model
    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    predicted_label = labels[class_index]

    return {"prediction": predicted_label}

# Run FastAPI in background thread
def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=start_fastapi, daemon=True).start()

# ============================
# 💻 FRONTEND - STREAMLIT APP
# ============================
st.set_page_config(page_title="🐾 Cat vs Dog Classifier", layout="centered")
st.title("🐶🐱 Cat vs Dog Classifier (FastAPI + Streamlit)")
st.write("Upload an image and let the AI decide if it’s a cat or dog!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify"):
        with st.spinner("Analyzing... please wait..."):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post("http://127.0.0.1:8000/predict/", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Prediction: {result['prediction']}")
                else:
                    st.error("❌ Error from backend")
            except Exception as e:
                st.error(f"⚠️ Cannot connect to backend: {e}")
