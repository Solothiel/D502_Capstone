
import pandas as pd
import pytest
from App.Save_Best_Model import select_best_model

def test_select_best_model_logic():
    df = pd.DataFrame({
        "Model": ["A", "B"],
        "Accuracy": [0.8, 0.9],
        "Precision": [0.7, 0.85],
        "Recall": [0.6, 0.75],
        "F1-Score": [0.63, 0.8],
        "ROC-AUC": [0.7,0.9]
    })

    trained_models = {"A": object(),"B": object()}
    name, model, score = select_best_model(df, trained_models)
    assert name == "B"
    assert score > 0
