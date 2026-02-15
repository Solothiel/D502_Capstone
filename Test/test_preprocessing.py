import pytest
import pandas as pd
from App.Preprocessing import preprocess_data

def test_preprocess_data_outputs():
    df = pd.DataFrame({
        "customer_id": [1,2],
        "occupation_status": ["Employed", "Unemployed"],
        "product_type": ["Type A", "Type B"],
        "loan_intent": ["Personal", "Business"],
        "loan_status": [1,0]
    })

    X,y = preprocess_data(df)
    assert X.shape[0] == y.shape[0] == 2
    assert "loan_status" not in X.columns
    assert y.tolist() == [1,0]