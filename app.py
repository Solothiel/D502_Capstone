import streamlit as st
import pandas as pd
import joblib
import os

# ------------------- Load Model and Artifacts -------------------

@st.cache_resource
def load_artifacts():
    # Load trained model
    model = joblib.load('data/best_model.pkl')
    # Load scaler
    scaler = joblib.load('data/scaler.pkl')
    # Load feature names
    feature_names = joblib.load("data/feature_names.pkl")
    # Load encoders
    categorical_cols = ['occupation_status', 'product_type', 'loan_intent']
    encoders = {col: joblib.load(f'data/{col}_encoder.pkl') for col in categorical_cols}

    return model, scaler, feature_names, encoders

best_model, scaler, feature_names, encoders = load_artifacts()

# ------------------- Streamlit UI -------------------

st.title("Loan Approval Prediction System")
st.write("Enter applicant information to predict loan approval.")

# Sidebar inputs
st.sidebar.header("Applicant Information")
customer_id = st.sidebar.text_input("Customer ID")
age = st.sidebar.number_input("Age", min_value=18, max_value=100)
annual_income = st.sidebar.number_input("Annual Income ($)", min_value=0.0)
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=0.0)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850)

occupation_status = st.sidebar.selectbox(
    "Occupation Status",
    encoders['occupation_status'].classes_
)
product_type = st.sidebar.selectbox(
    "Product Type",
    encoders['product_type'].classes_
)
loan_intent = st.sidebar.selectbox(
    "Loan Intent",
    encoders['loan_intent'].classes_
)

# ------------------- Predict Button -------------------

if st.button("Predict Loan Approval"):

    # Step 1: Create input DataFrame
    input_data = pd.DataFrame([{
        'customer_id': customer_id,
        'age': age,
        'annual_income': annual_income,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'occupation_status': encoders['occupation_status'].transform([occupation_status])[0],
        'product_type': encoders['product_type'].transform([product_type])[0],
        'loan_intent': encoders['loan_intent'].transform([loan_intent])[0]
    }])

    # Step 2: Drop customer_id and prepare features
    X_new = input_data.drop(columns=['customer_id'])

    # ------------------- Step 3: ALIGN FEATURES -------------------
    # Add missing columns with 0, remove extras, enforce correct order
    for col in feature_names:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[feature_names]
    # ---------------------------------------------------------------

    # Step 4: Scale features
    X_new_scaled = scaler.transform(X_new)

    # Step 5: Predict
    prediction = best_model.predict(X_new_scaled)[0]
    probability = best_model.predict_proba(X_new_scaled)[0][1]

    # Step 6: Display results
    st.subheader("Prediction Result")
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

# Footer
st.caption("Model trained and selected in Branch 1")