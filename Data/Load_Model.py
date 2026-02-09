import joblib

#loads everything from the other model.
best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

categorical_cols = ['occupation_status', 'product_type', 'loan_intent']
encoders = {col: joblib.load(f'{col}_encoder.pkl') for col in categorical_cols}

