import streamlit as st
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import uvicorn
import threading
import tempfile
from model import predict_cat_dog  # ✅ import your function

# ============================
# 🧠 BACKEND - FASTAPI APP
# ============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Run your model function
    prediction = predict_cat_dog(tmp_path)

    return {"prediction": prediction}

def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=start_fastapi, daemon=True).start()

# ============================
# 💻 FRONTEND - STREAMLIT APP
# ============================
st.set_page_config(page_title="🐾 Image Classifier", layout="centered")
st.title("🐾 Cat & Dog Classifier (FastAPI + Streamlit)")
st.write("Upload an image — the model will identify the breed or species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
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
