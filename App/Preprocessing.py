
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    categorical_cols = ["occupation_status", "product_type", "loan_intent"]
    df_encode = df.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encode[col] = le.fit_transform(df_encode[col])

    X = df_encode.drop(columns=["customer_id", "loan_status"])
    y = df_encode["loan_status"]

    return X, y



