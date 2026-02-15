import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_metrics(results_df: pd.DataFrame, metrics=None):
    if metrics is None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    for metric in metrics:
        plt.figure(figsize=(8, 4))
        sns.barplot(x="Model", y=metric, data=results_df)
        plt.title(f'{metric} Comparison')
        plt.show()