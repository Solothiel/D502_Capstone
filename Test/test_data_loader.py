import pytest
import pandas as pd
from App.Data_file import load_data, preview_data


def test_load_data_returns_datafram():
    df = load_data("Data/loan_approval_data_2025.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "loan_status" in df.columns

def test_preview_dat_runs(capsys):
    df = pd.DataFrame({
        "loan_status": [1,0,1],
        "customer_id": [101,102,103]
    })
    preview_data(df)
    captured = capsys.readouterr()
    assert "loan_status" in captured.out

