import streamlit as st
from model import predict_cat_dog
from PIL import Image
import tempfile

st.set_page_config(page_title="🐾 Image Classifier", layout="centered")
st.title("🐾 Cat & Dog Classifier")
st.write("Upload an image — the model will identify the breed or species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing... please wait..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            result = predict_cat_dog(tmp_path)
            st.success(f"✅ Prediction: {result}")
