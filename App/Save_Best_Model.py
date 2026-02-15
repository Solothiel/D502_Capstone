import pandas as pd

from Model_Evaluator import results_df
from Train_models import trained_model

#define weights for loan approval

weights = {
    "F1-Score": 0.25,   # balance metric
    "Accuracy": 0.15,   # general correctness
    "Precision": 0.35,  # avoid bad loan approvals
    "Recall": 0.15,
    "ROC-AUC": 0.10
}
#compute overall weighted score for each model
results_df['Overall_score'] = sum(results_df[m] * w for m, w in weights.items())

#selects the best model
best_model_name = results_df.sort_values("Overall_score", ascending = False).iloc[0]["Model"]
best_model = trained_model[best_model_name]


#return results
overall_score = results_df.loc[results_df['Model'] == best_model_name, "Overall_score"].values[0]

print(f"Best Model for loan approval: {best_model_name} with overall score {overall_score:.4f}")
