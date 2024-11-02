# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import yaml

# Load model and parameters
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

model = joblib.load(params['output']['model'])

# Define the input data model using Pydantic
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

# Initialize FastAPI app
app = FastAPI(title="Diabetes Prediction API", description="An API to predict diabetes using a trained Random Forest model")

# Define the prediction endpoint
@app.post("/predict")
def predict_diabetes(input_data: DiabetesPredictionInput):
    # Convert the input data into a DataFrame for model compatibility
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make predictions
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0, 1]
    
    # Prepare response
    result = {
        "prediction": int(prediction[0]),  # 0 or 1 for diabetes status
        "probability": probability  # Probability of having diabetes
    }
    return result

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Diabetes Prediction API. Use the /predict endpoint to make predictions."}
