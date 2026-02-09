import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and encoders


def load_artifacts():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')

    categorical_cols = ['occupation_status', 'product_type', 'loan_intent']
    encoders = {col: joblib.load(f'{col}_encoder.pkl') for col in categorical_cols}

    return model, scaler, encoders

best_model, scaler, encoders = load_artifacts()

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

# Predict button
if st.button("Predict Loan Approval"):

    # Create input DataFrame
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

    # Prepare features
    X_new = input_data.drop(columns=['customer_id'])
    X_new_scaled = scaler.transform(X_new)

    # Predict
    prediction = best_model.predict(X_new_scaled)[0]
    probability = best_model.predict_proba(X_new_scaled)[0][1]

    # Display results
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
