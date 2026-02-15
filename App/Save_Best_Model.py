import pandas as pd


def select_best_model(results_df: pd.DataFrame, trained_models: dict, weights=None):
    if weights is None:
        weights = {
            "F1-Score": 0.25,
            "Accuracy": 0.15,
            "Precision": 0.35,
            "Recall": 0.15,
            "ROC-AUC": 0.10
        }

    results_df['Overall_score'] = sum(results_df[m] * w for m, w in weights.items())
    best_model_name = results_df.sort_values("Overall_score", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    overall_score = results_df.loc[results_df['Model'] == best_model_name, "Overall_score"].values[0]

    return best_model_name, best_model, overall_score