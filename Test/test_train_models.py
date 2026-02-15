import pytest
import pandas as pd
from App.Train_models import get_models, train_models
def test_get_models_returns_dict():
    models = get_models()
    assert isinstance(models, dict)
    assert "Logistic Regression" in models

def test_train_models_outputs():
    models = get_models()
    X_train = pd.DataFrame({"f1": [1,2], "f2":[3,4]})
    y_train = pd.Series([0,1])
    trained = train_models(models, X_train, y_train)
    for model in trained.values():
        assert hasattr(model, "predict")