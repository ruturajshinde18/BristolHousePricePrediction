from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import Optional
import uvicorn

# Load the trained model once at startup
try:
    with open('../models/catboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = FastAPI(title="Bristol House Price Predictor API", version="1.0.0")


class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    property_type: str
    new_build: str = "N"  # Default to "N" (not new build)
    tenure: str = "F"  # Default to "F" (freehold)
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
        # Prepare input data in the exact format the model expects
        input_data = pd.DataFrame({
            'property_type': [request.property_type],
            'new_build': [request.new_build],
            'tenure': [request.tenure],
            'Year': [request.year],
            'lat': [request.latitude],
            'long': [request.longitude]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Format the prediction
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
    uvicorn.run(app, host="0.0.0.0", port=8000)