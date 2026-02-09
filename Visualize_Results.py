import matplotlib.pyplot as plt
import seaborn as sns

from Model_Evaluator import results_df

metrics = ['Accuracy', "Precision", "Recall", "F1-Score", "ROC-AUC"]

for metric in metrics:
    plt.figure(figsize=(8,4))
    sns.barplot(x="Model", y=metric, data=results_df)
    plt.title(f'{metric} Comparison')
    plt.show()