import streamlit as st
import pandas as pd
import joblib

# Load model and baseline dummy row
@st.cache_resource
def load_files():
    model = joblib.load('models/home_credit_xgb_pipeline.joblib')
    baseline = joblib.load('models/baseline_features.joblib')
    return model, baseline

model, baseline_df = load_files()

st.title("🏦 Credit Default Risk Predictor")
st.write("Enter applicant details below to calculate loan default probability.")

# User inputs
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income ($)", min_value=10000, max_value=2000000, value=150000, step=5000)
    credit = st.number_input("Loan Amount ($)", min_value=10000, max_value=3000000, value=500000, step=10000)
    annuity = st.number_input("Loan Annuity ($)", min_value=1000, max_value=200000, value=25000, step=1000)
    age = st.slider("Age (Years)", 18, 75, 35)

with col2:
    emp_years = st.slider("Years Employed", 0, 45, 5)
    ext1 = st.slider("External Credit Score 1", 0.0, 1.0, 0.50)
    ext2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.50)
    ext3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.50)

if st.button("Predict Default Risk", type="primary"):
    input_df = baseline_df.copy()

    # Update baseline row with user inputs (only updates if column exists in your X_train)
    updates = {
        'AMT_INCOME_TOTAL': income,
        'AMT_CREDIT': credit,
        'AMT_ANNUITY': annuity,
        'DAYS_BIRTH': -age * 365,
        'DAYS_EMPLOYED': -emp_years * 365,
        'EXT_SOURCE_1': ext1,
        'EXT_SOURCE_2': ext2,
        'EXT_SOURCE_3': ext3,
    }

    for col, val in updates.items():
        if col in input_df.columns:
            input_df[col] = val

    # Predict
    prob = model.predict_proba(input_df)[0][1]

    st.divider()
    st.metric("Default Probability", f"{prob:.2%}")

    if prob < 0.15:
        st.success("✅ Low Risk — Approve Loan")
    elif prob < 0.35:
        st.warning("⚠️ Moderate Risk — Manual Review Needed")
    else:
        st.error("🚨 High Risk — Reject / High Default Chance")