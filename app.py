import streamlit as st
import pandas as pd
import joblib
import os

# ------------------- Setup paths -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ------------------- Load Model and Artifacts -------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(DATA_DIR, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(DATA_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(DATA_DIR, "feature_names.pkl"))
    feature_means = joblib.load(os.path.join(DATA_DIR, "feature_means.pkl"))
    categorical_cols = ['occupation_status', 'product_type', 'loan_intent']
    encoders = {col: joblib.load(os.path.join(DATA_DIR, f'{col}_encoder.pkl')) for col in categorical_cols}
    return model, scaler, feature_names, feature_means, encoders

best_model, scaler, feature_names, feature_means, encoders = load_artifacts()

# ------------------- Streamlit UI -------------------
st.title("Loan Approval Prediction System")
st.write("Enter applicant information to predict loan approval.")

# Sidebar inputs
st.sidebar.header("Applicant Information")
customer_id = st.sidebar.text_input("Customer ID")
age = st.sidebar.number_input("Age", min_value=18, max_value=100)
annual_income = st.sidebar.number_input("Annual Income ($)", min_value=1.0)  # cannot be zero
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=0.0)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850)
current_debt = st.sidebar.number_input("Current Debt ($)", min_value=0.0)
payment = st.sidebar.number_input("Monthly Payment ($)", min_value=0.0)
occupation_status = st.sidebar.selectbox("Occupation Status", encoders['occupation_status'].classes_)
product_type = st.sidebar.selectbox("Product Type", encoders['product_type'].classes_)
loan_intent = st.sidebar.selectbox("Loan Intent", encoders['loan_intent'].classes_)

# ------------------- Predict Button -------------------
if st.button("Predict Loan Approval"):

    # Compute ratios from user input
    debt_to_income_ratio = current_debt / annual_income
    loan_to_income_ratio = loan_amount / annual_income
    payment_to_income_ratio = payment / annual_income

    # Warn if income is too low
    if annual_income < 1000:
        st.warning("Annual income is very low – model may decline loan.")

    # Step 1: Create input DataFrame
    input_data = pd.DataFrame([{
        'customer_id': customer_id,
        'age': float(age),
        'annual_income': float(annual_income),
        'loan_amount': float(loan_amount),
        'credit_score': float(credit_score),
        'current_debt': float(current_debt),
        'debt_to_income_ratio': float(debt_to_income_ratio),
        'loan_to_income_ratio': float(loan_to_income_ratio),
        'payment_to_income_ratio': float(payment_to_income_ratio),
        'occupation_status': int(encoders['occupation_status'].transform([occupation_status])[0]),
        'product_type': int(encoders['product_type'].transform([product_type])[0]),
        'loan_intent': int(encoders['loan_intent'].transform([loan_intent])[0])
    }])

    # Step 2: Drop customer_id and align features
    X_new = input_data.drop(columns=['customer_id'])
    for col in feature_names:
        if col not in X_new.columns:
            X_new[col] = float(feature_means.get(col, 0))
    X_new = X_new[feature_names]

    # Step 3: Scale features
    X_new_scaled = scaler.transform(X_new)

    # Step 4: Debug output
    st.subheader("Debug - Features sent to model")
    for i, col in enumerate(feature_names):
        st.write(f"{i+1}. {col}: {X_new.iloc[0][col]}")
    st.subheader("Debug - Scaled features")
    st.dataframe(pd.DataFrame(X_new_scaled, columns=feature_names))

    # Step 5: Predict
    prediction = best_model.predict(X_new_scaled)[0]
    probability_all = best_model.predict_proba(X_new_scaled)[0]
    probability = probability_all[1]  # probability of approval

    st.subheader("Prediction Result")
    st.write(f"Class probabilities (Not Approved / Approved): {probability_all}")
    if prediction == 1:
        st.success(f"✅ Loan Approved (Probability: {probability:.2%})")
    else:
        st.error(f"❌ Loan Not Approved (Probability: {probability:.2%})")

    st.write("Model Output:")
    st.dataframe(
        input_data.assign(
            predicted_approval=prediction,
            approval_probability=probability
        )
    )

st.caption("Model trained and selected in Branch 1")
