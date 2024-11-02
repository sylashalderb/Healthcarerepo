# Correcting the feature names to match the model's expectations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import logging
import uvicorn
from typing import Dict
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes using machine learning model",
    version="1.0.0"
)

# Load the model at startup
try:
    model_path = os.path.join('models', 'rf_model.joblib')
    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError(f"Failed to load model from {model_path}")

class DiabetesPredictionInput(BaseModel):
    age: float
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    hypertension: int
    heart_disease: int
    gender_Female: int
    gender_Male: int
    gender_Other: int
    smoking_history_current: int
    smoking_history_ever: int
    smoking_history_former: int
    smoking_history_never: int
    smoking_history_not_current: int

    class Config:
        schema_extra = {
            "example": {
                "age": 50.0,
                "bmi": 28.5,
                "HbA1c_level": 7.0,
                "blood_glucose_level": 150.0,
                "hypertension": 1,
                "heart_disease": 0,
                "gender_Female": 0,
                "gender_Male": 1,
                "gender_Other": 0,
                "smoking_history_current": 0,
                "smoking_history_ever": 1,
                "smoking_history_former": 0,
                "smoking_history_never": 0,
                "smoking_history_not_current": 0
            }
        }

@app.get("/", tags=["Health Check"])
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Diabetes Prediction API is running"}

@app.post("/predict", response_model=Dict)
async def predict_diabetes(input_data: DiabetesPredictionInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Ensure feature names match those used during model training
        features = [
            'age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
            'hypertension', 'heart_disease', 'gender_Female', 'gender_Male',
            'gender_Other', 'smoking_history_current', 'smoking_history_ever',
            'smoking_history_former', 'smoking_history_never',
            'smoking_history_not_current'
        ]

        # Prepare final input
        final_input = input_df[features]

        # Make prediction
        prediction = model.predict(final_input)
        probability = model.predict_proba(final_input)[0, 1]

        # Prepare response
        response = {
            "status": "success",
            "prediction": int(prediction[0]),
            "probability": float(probability),
            "message": "Diabetes" if prediction[0] == 1 else "No Diabetes"
        }

        logger.info(f"Successful prediction made: {response}")
        return response

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"status": "error", "message": str(exc)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)