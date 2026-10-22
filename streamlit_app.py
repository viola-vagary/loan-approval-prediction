import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Title & Basic Config
st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")

# Clean, professional styling with custom sidebar and modern gradient
st.markdown("""
    <style>
    /* Modern, vibrant blue and pink gradient on the right side / main background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: linear-gradient(135deg, #DBEAFE 0%, #E0E7FF 35%, #FCE7F3 70%, #FBCFE8 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Hide Streamlit default Enter tooltip */
    div[data-testid="InputInstructions"] {
        display: none !important;
    }

    /* Hide Streamlit Deploy button, 3 dots (MainMenu), and footer */
    .stDeployButton,
    div[data-testid="stToolbarActions"],
    #MainMenu,
    footer,
    header [data-testid="stToolbar"] {
        visibility: hidden !important;
        display: none !important;
    }
    
    /* Main Page Headings - Deep Navy Blue for high contrast */
    h1 {
        color: #1E3A8A !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Subtitle text */
    .stApp p, .stApp span {
        color: #1E293B;
    }
    
    /* Frosted-glass semi-transparent card for Login */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.72) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        padding: 2.2rem !important;
        box-shadow: 0 12px 32px 0 rgba(31, 38, 135, 0.12) !important;
    }

    /* Sub-section headers inside form */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E3A8A !important;
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid #DBEAFE;
    }

    /* Column Headers inside the form */
    .column-header {
        font-size: 1rem;
        font-weight: 700;
        color: #1E40AF !important;
        margin-bottom: 0.8rem;
    }

    /* ALL Input Labels inside the form: Deep Charcoal/Slate for 100% Readability */
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p {
        font-weight: 600 !important;
        color: #0F172A !important;
        font-size: 0.95rem !important;
    }

    /* Input text boxes & select boxes: Clean white background with dark readable text */
    div[data-baseweb="input"], 
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        border: 1.5px solid #CBD5E1 !important;
    }
    div[data-baseweb="input"] input {
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="select"] span {
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="input"]:focus-within, 
    div[data-baseweb="select"]:focus-within {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
    }

    /* Green Login Button (never turns white on click/hover) */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #16A34A !important;
        border: none !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 10px rgba(22, 163, 74, 0.25) !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #15803D !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stFormSubmitButton"] > button:active,
    div[data-testid="stFormSubmitButton"] > button:focus {
        background-color: #166534 !important;
        color: #FFFFFF !important;
    }

    /* Primary Prediction Button */
    button[kind="primary"] {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(90deg, #1D4ED8 0%, #1E40AF 100%) !important;
    }

    /* Modern Dark Blue Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0F1D 0%, #1E3A8A 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] button {
        background-color: #EF4444 !important;
        border: none !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #DC2626 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Loan Approval Prediction System")
st.write("A machine learning model to predict loan application approval status.")

# 2. Load Model and Scaler
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base_dir, 'loan_status_predictor.pkl'))
    scaler = joblib.load(os.path.join(base_dir, 'vector.pkl'))
    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 3. Simple Login Simulation
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Demo credentials (clean, unique passwords to avoid browser breach warnings)
USERS = {
    "admin": "ApexAdm#2026",
    "officer": "LoanOff#8821"
}

if not st.session_state['logged_in']:
    col_left, col_center, col_right = st.columns([0.4, 2.2, 0.4])
    with col_center:
        st.subheader("🔐 Staff Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            st.write("")
            login_btn = st.form_submit_button("Login", use_container_width=True)

            if login_btn:
                if username in USERS and USERS[username] == password:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    st.stop()

# Sidebar for logout
with st.sidebar:
    st.write(f"Logged in as: **{st.session_state['username']}**")
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

# 4. Input Form
st.write("")
with st.form("loan_application_form"):
    st.markdown('<div class="section-title">📝 Application Overview & Metrics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="column-header">👤 Personal Profile</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    with col2:
        st.markdown('<div class="column-header">💰 Financial & Loan Metrics</div>', unsafe_allow_html=True)
        applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, value=4500.0, step=100.0)
        coapplicant_income = st.number_input("Co-Applicant Income ($)", min_value=0.0, value=1500.0, step=100.0)
        loan_amount = st.number_input("Loan Amount (Thousands $)", min_value=1.0, value=120.0, step=5.0)
        term_options = {
            "360 Months (30 Years)": 360.0,
            "240 Months (20 Years)": 240.0,
            "180 Months (15 Years)": 180.0,
            "120 Months (10 Years)": 120.0,
            "84 Months (7 Years)": 84.0,
            "60 Months (5 Years)": 60.0
        }
        selected_term = st.selectbox("Loan Term", list(term_options.keys()), index=0)
        loan_term = term_options[selected_term]
        credit_history = st.selectbox("Credit History", ["1.0 (Good)", "0.0 (Poor)"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.write("")
    predict_clicked = st.form_submit_button("🔍 Predict Loan Status", use_container_width=True)

# 5. Prediction
if predict_clicked:
    # Encoding map matching training notebook
    encoding = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 4},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Rural': 0, 'Semiurban': 2, 'Urban': 1},
        'Credit_History': {'1.0 (Good)': 1.0, '0.0 (Poor)': 0.0}
    }

    # Prepare DataFrame
    input_data = pd.DataFrame([{
        'Gender': encoding['Gender'][gender],
        'Married': encoding['Married'][married],
        'Dependents': encoding['Dependents'][dependents],
        'Education': encoding['Education'][education],
        'Self_Employed': encoding['Self_Employed'][self_employed],
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': float(loan_term),
        'Credit_History': encoding['Credit_History'][credit_history],
        'Property_Area': encoding['Property_Area'][property_area]
    }])

    # Scale numeric features
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Predict
    prediction = model.predict(input_data)[0]
    raw_probs = model.predict_proba(input_data)[0]
    total = np.sum(raw_probs)
    approval_prob = (raw_probs[1] / total) * 100 if total > 0 else 50.0

    st.write("")
    if prediction == 1:
        st.markdown(f"""
            <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 1.25rem 1.5rem; margin-top: 1rem;">
                <h3 style="color: #065F46; margin: 0 0 0.4rem 0;">🎉 Loan Approved!</h3>
                <p style="color: #047857; margin: 0; font-size: 0.95rem;">
                    The applicant meets the eligibility standards with an approval confidence of <strong>{approval_prob:.1f}%</strong>.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 12px; padding: 1.25rem 1.5rem; margin-top: 1rem;">
                <h3 style="color: #991B1B; margin: 0 0 0.4rem 0;">❌ Loan Not Approved</h3>
                <p style="color: #B91C1C; margin: 0; font-size: 0.95rem;">
                    The applicant does not currently meet credit/underwriting criteria (Risk Confidence: <strong>{100 - approval_prob:.1f}%</strong>).
                </p>
            </div>
        """, unsafe_allow_html=True)
