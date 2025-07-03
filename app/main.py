from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os
import subprocess
from typing import Optional
import uvicorn

MODEL_PATH = "models/catboost_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Pulling with DVC...")
        try:
            subprocess.run(["dvc", "pull"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"DVC pull failed: {e}")
            return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

app = FastAPI(title="Bristol House Price Predictor API", version="1.0.0")

class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    property_type: str
    new_build: str = "N"
    tenure: str = "F"
    year: Optional[int] = 2024

class PredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str
    location: dict
    inputs_used: dict

@app.get("/")
async def root():
    return {
        "message": "Bristol House Price Predictor API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_data = pd.DataFrame({
            'property_type': [request.property_type],
            'new_build': [request.new_build],
            'tenure': [request.tenure],
            'Year': [request.year],
            'lat': [request.latitude],
            'long': [request.longitude]
        })

        prediction = model.predict(input_data)[0]
        formatted_price = f"£{prediction:,.0f}"

        return PredictionResponse(
            predicted_price=float(prediction),
            formatted_price=formatted_price,
            location={
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            inputs_used={
                "property_type": request.property_type,
                "new_build": request.new_build,
                "tenure": request.tenure,
                "year": request.year,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_status": "loaded" if model else "not_loaded"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9002)
