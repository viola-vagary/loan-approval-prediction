# 🏦 Loan Approval Prediction System

An end-to-end Machine Learning web application and REST API for predicting loan approval eligibility based on applicant financial profiles.

---

## 🚀 Features

- **Interactive Streamlit Web UI**: Easy-to-use form with dropdowns, sliders, real-time approval percentage, and visual alerts.
- **FastAPI REST Service**: Production-ready API accepting human-readable JSON payloads (`"Male"`, `"Graduate"`, etc.) with automatic data validation and probability scoring.
- **Pretrained ML Model**: Scikit-Learn Random Forest Classifier trained on applicant financial demographics.

---

## 🛠️ Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 2. Run the Streamlit Web Application (Recommended)
Launch the visual web interface in your browser:
```bash
streamlit run streamlit_app.py
```
> The app will automatically open in your default browser at `http://localhost:8501`.

---

### 3. Run the FastAPI REST API
To serve the prediction endpoint via REST API:
```bash
uvicorn app:app --reload
```
- Open Swagger UI for interactive testing: **`http://127.0.0.1:8000/docs`**
- Test endpoint: `POST http://127.0.0.1:8000/predict`

#### Sample JSON Payload:
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 4583.0,
  "CoapplicantIncome": 1508.0,
  "LoanAmount": 128.0,
  "Loan_Amount_Term": 360.0,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

#### Sample Response:
```json
{
  "loan_status": "Approved",
  "approval_probability": 86.45,
  "rejection_probability": 13.55
}
```

---

## 📁 Repository Structure
```text
├── app.py                      # FastAPI REST application
├── streamlit_app.py            # Streamlit interactive Web UI
├── loan_status_predictor.pkl   # Serialized Random Forest model
├── vector.pkl                  # Serialized feature scaler (StandardScaler)
├── loan_data.csv               # Training dataset
├── Loan Approval Prediction.ipynb # Data analysis & model training notebook
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```