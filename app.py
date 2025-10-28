import streamlit as st
import requests

st.title("Cat vs Dog Prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Send image to FastAPI backend
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)
    
    st.write("Prediction:", response.json()["prediction"])
