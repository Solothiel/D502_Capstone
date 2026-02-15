from App.Data_file import load_data, preview_data
from App.Preprocessing import preprocess_data
from App.Split_Data import split_and_scale
from App.Train_models import get_models, train_models
from App.Model_Evaluator import evaluate_models
from App.Visualize_Results import plot_metrics
from App.Save_Best_Model import select_best_model

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def main():
    # Load dataset
    data_path = "Data/Loan_approval_data_2025.csv"
    df = load_data(data_path)
    preview_data(df)

    # Preprocess data
    X, y = preprocess_data(df)

    # Split and scale
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_means = split_and_scale(X, y)

    # baseline model for Hypothesis
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X_train_scaled, y_train)
    y_baseline = baseline_model.predict(X_test_scaled)
    baseline_acc = accuracy_score(y_test, y_baseline)
    print(f"\nBaseline accuracy (majority class) = {baseline_acc:.2f}")


    # Initialize and train models
    models = get_models()
    trained_models = train_models(models, X_train_scaled, y_train)  # all models use scaled features for simplicity

    # Evaluate models
    results_df = evaluate_models(
        trained_models,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X_train_raw=X_train_scaled,
        X_test_raw=X_test_scaled
    )
    print("\nModel Evaluation Results:")
    print(results_df)

    # Compares each model to the baseline
    print("\nHypothesis Testing: Model vs Baseline")
    for name, model in trained_models.items():
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        print(f"{name} -- Accuracy: {acc:.2f}, F1: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")

        # McNemar test
        tb = np.zeros((2, 2))
        for a, b in zip(y_pred, y_baseline):
            tb[a, b] += 1
        result = mcnemar(tb, exact=True)
        sig = "Reject H0" if result.pvalue < 0.05 else "Fail to reject H0"
        print(f"McNemar test p-value={result.pvalue:.4f} → {sig}\n")

    #  Visualize metrics
    plot_metrics(results_df)

    # Feature importance (evidence for H₁)
    if "Random Forest" in trained_models:
        rf = trained_models["Random Forest"]
        importances = rf.feature_importances_
        df_importance = pd.DataFrame({"Feature": X_train.columns, "Importance": importances})
        df_importance = df_importance.sort_values(by="Importance", ascending=False)
        print("\nFeature Importance (Random Forest):")
        print(df_importance)

        # Plot feature importance
        df_importance.plot(kind="bar", x="Feature", y="Importance", legend=False)
        plt.title("Random Forest Feature Importance")
        plt.show()

    # 7Select best model using weighted scoring
    best_model_name, best_model, overall_score = select_best_model(results_df, trained_models)
    print(f"\nBest Model for Loan Approval: {best_model_name} with overall score {overall_score:.4f}")

if __name__ == "__main__":
    main()
