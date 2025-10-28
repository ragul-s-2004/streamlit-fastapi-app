import streamlit as st
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import uvicorn
import threading

# ============================
# 🧠 BACKEND - FASTAPI APP
# ============================
app = FastAPI()

# Allow Streamlit frontend to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy model logic (replace with your classifier)
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    # Here you can load your model and predict
    # Example dummy prediction:
    return {"prediction": "This looks like a Cat 🐱"}

# Function to start FastAPI in a thread
def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start FastAPI server in a background thread
threading.Thread(target=start_fastapi, daemon=True).start()

# ============================
# 💻 FRONTEND - STREAMLIT APP
# ============================
st.set_page_config(page_title="Image Classifier", layout="centered")

st.title("🧠 Simple Image Classifier (FastAPI + Streamlit)")
st.write("Upload an image and get a prediction!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        with st.spinner("Predicting..."):
            # Send to FastAPI
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
