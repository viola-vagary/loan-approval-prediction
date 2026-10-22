from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib 
import pandas as pd
import os

app = FastAPI(
    title="Loan Approval Prediction API",
    description="API for evaluating loan applications using machine learning."
)

base_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_dir, 'loan_status_predictor.pkl'))
scaler = joblib.load(os.path.join(base_dir, 'vector.pkl'))

num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

class LoanApproval(BaseModel):
    Gender: Literal["Male", "Female"] = Field(..., example="Male")
    Married: Literal["Yes", "No"] = Field(..., example="Yes")
    Dependents: Literal["0", "1", "2", "3+"] = Field(..., example="1")
    Education: Literal["Graduate", "Not Graduate"] = Field(..., example="Graduate")
    Self_Employed: Literal["Yes", "No"] = Field(..., example="No")
    ApplicantIncome: float = Field(..., ge=0, example=4583.0)
    CoapplicantIncome: float = Field(..., ge=0, example=1508.0)
    LoanAmount: float = Field(..., gt=0, example=128.0)
    Loan_Amount_Term: float = Field(..., gt=0, example=360.0)
    Credit_History: Literal[0, 1] = Field(..., example=1, description="1 if credit guidelines are met, 0 otherwise")
    Property_Area: Literal["Rural", "Semiurban", "Urban"] = Field(..., example="Urban")

ENCODINGS = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 4},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 2, 'Urban': 1}
}

@app.get("/")
def home():
    return {"message": "Loan Approval Prediction API is live. Visit /docs for Swagger documentation."}

@app.post("/predict")
async def predict_loan_status(application: LoanApproval):
    try:
        data = application.dict()
        
        # Apply human-readable string to model-encoding mapping
        data['Gender'] = ENCODINGS['Gender'][data['Gender']]
        data['Married'] = ENCODINGS['Married'][data['Married']]
        data['Dependents'] = ENCODINGS['Dependents'][data['Dependents']]
        data['Education'] = ENCODINGS['Education'][data['Education']]
        data['Self_Employed'] = ENCODINGS['Self_Employed'][data['Self_Employed']]
        data['Property_Area'] = ENCODINGS['Property_Area'][data['Property_Area']]
        data['Credit_History'] = float(data['Credit_History'])

        input_data = pd.DataFrame([data])
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        prediction = model.predict(input_data)[0]
        raw_probs = model.predict_proba(input_data)[0]
        total = float(raw_probs.sum())
        norm_probs = raw_probs / total if total > 0 else [0.5, 0.5]

        return {
            'loan_status': "Approved" if prediction == 1 else "Not Approved",
            'approval_probability': round(float(norm_probs[1]) * 100, 2),
            'rejection_probability': round(float(norm_probs[0]) * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))