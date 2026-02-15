import pytest
import pandas as pd
from App.Model_Evaluator import evaluate_models
from App.Train_models import get_models, train_models

def test_evaluate_models_dataframe():
    X_train = pd.DataFrame({"f1": [1,2], "f2": [3,4]})
    X_test = pd.DataFrame({"f1": [5,6], "f2": [7,8]})
    y_train = pd.Series([0,1])
    y_test = pd.Series([1,0])

    models = get_models()
    trained_models = train_models(models,X_train, y_train)

    df_results = evaluate_models(
        trained_models,
        X_train_scaled=X_train,
        X_test_scaled=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_raw=X_train,
        X_test_raw=X_test
    )
    assert isinstance(df_results, pd.DataFrame)
    assert all(col in df_results.columns for col in ["Model","Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
