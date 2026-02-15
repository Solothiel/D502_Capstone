import pytest
import pandas as pd
from App.Split_Data import split_and_scale

def test_split_and_scale_shapes():
    X = pd.DataFrame({"feature1": [1,2,3,4], "feature2": [5,6,7,8]})
    y = pd.Series([0,1,0,1])
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_means = split_and_scale(X, y, test_size=0.5)
    assert X_train.shape[0] == y_train.shape[0] == 2
    assert X_test.shape[0] == y_test.shape[0] == 2
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert isinstance(feature_means, dict)
