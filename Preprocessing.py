import joblib
from sklearn.preprocessing import LabelEncoder
from Data_file import loan_df


#Encode categorical variables
categorical_cols = ["occupation_status", 'product_type', 'loan_intent']
df_encode = loan_df.copy()

#Encode each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    df_encode[col] = le.fit_transform(df_encode[col])

    joblib.dump(le, f"data/{col}_encoder.pk1")

X = df_encode.drop(columns={'customer_id', 'loan_status'})
y = df_encode['loan_status']

