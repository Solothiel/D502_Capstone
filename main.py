from App.Data_file import load_data, preview_data
from App.Preprocessing import preprocess_data
from App.Split_Data import split_and_scale
from App.Train_models import get_models, train_models
from App.Model_Evaluator import evaluate_models
from App.Visualize_Results import plot_metrics
from App.Save_Best_Model import select_best_model

def main():
    # Load dataset
    data_path = "Data/Loan_approval_data_2025.csv"
    df = load_data(data_path)
    preview_data(df)

    # Preprocess data
    X, y = preprocess_data(df)

    # Split and scale
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_means = split_and_scale(X, y)

    # Initialize and train models
    models = get_models()
    trained_models = train_models(models, X_train_scaled, y_train)  # all models use scaled features for simplicity

    # Evaluate models
    results_df = evaluate_models(
        trained_models,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        X_train_raw=X_train,
        X_test_raw=X_test
    )
    print("\nModel Evaluation Results:")
    print(results_df)

    #  Visualize metrics
    plot_metrics(results_df)

    # 7Select best model using weighted scoring
    best_model_name, best_model, overall_score = select_best_model(results_df, trained_models)
    print(f"\nBest Model for Loan Approval: {best_model_name} with overall score {overall_score:.4f}")

if __name__ == "__main__":
    main()
