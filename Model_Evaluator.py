import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from Split_Data import X_train_scale, X_test_scale, y_train, X_train, X_test, y_test
from Train_models import models, trained_model

results = []

for name, model in models.items():
    #logistic Regression uses scaled data, others use raw.
    if name == "Logistic Regression":
        model.fit(X_train_scale, y_train)
        y_pred = model.predict(X_test_scale)
        y_prob = model.predict_proba(X_test_scale)[:,1]

    else:
        model.fit(X_train, y_train)
        y_pred =model.predict(X_test)
        y_prob =model.predict_proba(X_test)[:,1]

    # Saved trained model
    trained_model[name] = model

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append([name, acc, prec, rec, f1, roc_auc])

# enter results in a dataframe.

results_df =pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
print(results_df)