import pytest
import pandas as pd
from App.Visualize_Results import plot_metrics

def test_plot_metrics_runs():
    df = pd.DataFrame({
        "Model": ["A", "B"],
        "Accuracy": [0.8, 0.9],
        "Precision": [0.7, 0.85],
        "Recall": [0.6, 0.75],
        "F1-Score": [0.65, 0.8],
        "ROC-AUC": [0.7, 0.9]
    })
    plot_metrics(df)  # ensure it runs without error
