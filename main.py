from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from model import predict_cat_dog  # your function from Step 1

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    with open("temp.jpg", "wb") as f:
        f.write(await file.read())

    # Get prediction
    result = predict_cat_dog("temp.jpg")
    return JSONResponse({"prediction": result})
