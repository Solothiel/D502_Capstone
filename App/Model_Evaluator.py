import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_models(models, X_train_scaled, X_test_scaled, y_train, y_test, X_train_raw=None, X_test_raw=None):
    results = []

    for name, model in models.items():
        if name == "Logistic Regression":
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(X_test_raw)
            y_prob = model.predict_proba(X_test_raw)[:, 1]

        results.append([
            name,
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, y_prob)
        ])

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
    return results_df
