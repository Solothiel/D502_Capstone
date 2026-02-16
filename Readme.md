# Loan Approval Prediction Project

## Project Overview

This project predicts whether a loan application will be approved based on 
customer features such as:

* Occupation status
* Product type
* Loan intent
* Other demographic and financial variables


The project compares multiple machine learning models (Logistic 
Regression, Decision Tree, Random Forest) and selects the best model using a weighted scoring 
system based on metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Null Hypothesis

### H₀ (Null Hypothesis):

    Applicant features — such as occupation, loan intent, product type, and other 
    variables — have no effect on loan approval. A model trained on these 
    features would perform no better than random guessing or a baseline classifier.

### H₁ (Alternative Hypothesis):

    At least one feature significantly influences loan approval, allowing predictive models to 
    perform better than baseline.



## Project Pipeline

1. Data Loading (`data_loader.py`)
   * Loads the loan dataset (`Loan_approval_data_2025.csv`)
   * Provides a preview of data, column info, and target distribution

2. Preprocessing (`preprocessing.py`)
   * Encodes categorical features using LabelEncoder
   * Creates feature matrix X and target y

3. Data Splitting & Scaling (`split_data.py`)
   * Splits dataset into train/test (80/20)
   * Standardizes features using StandardScaler

4. Model Training  (`train_models.py`)
   * Defines Logistic Regression, Decision Tree, Random Forest
   * Trains models on scaled training data

5. Model Evaluation (`model_evaluator.py`)
   * Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC
   * Evaluates all models on consistent scaled arrays

6. Visualization (`visualize_results.py`)
   * Generates bar plots comparing metrics across models

7. Best Model Selection (`select_best_model.py`)
   * Computes weighted overall score across metrics
   * Selects the best-performing model


## How to Run

1. Clone the repository:
    
`bash`

    `git clone <>`
    `cd loan_project`

2. Install dependencies:

`bash`

    `pip install -r requirements.txt`

3. Run the Pipeline:

`bash`

    `python D502_capstone/main.py`

4. Optional: Run tests with pytest:

`bash`

    `pytest tests/`



## Dependencies

* Python~= 3.13
* pandas~=2.3.3
* scikit-learn~=1.8.0
* matplotlib~=3.10.8
* seaborn~=0.13.2
* pytest~=9.0.2
      

## Project Structure 
```
loan_project/

├── app/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── train_models.py
│   ├── model_evaluator.py
│   ├── visualize_results.py
│   └── select_best_model.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_split_data.py
│   ├── test_train_models.py
│   ├── test_model_evaluator.py
│   ├── test_visualize_results.py
│   └── test_select_best_model.py
├── Data/
│   └── Loan_approval_data_2025.csv
├── main.py
├── requirements.txt
└── README.md
```

### Model Evaluation Results 
```
Model Evaluation Results:
                 Model  Accuracy  Precision    Recall  F1-Score   ROC-AUC
0  Logistic Regression    0.8463   0.851404  0.872337  0.861743  0.929275
1        Decision Tree    0.8749   0.885455  0.886906  0.886180  0.873593
2        Random Forest    0.9163   0.923399  0.924240  0.923819  0.974367

```

### Hypothesis Testing: Model vs. Baseline
```
Hypothesis Testing: Model vs Baseline
Logistic Regression -- Accuracy: 0.85, F1: 0.86, ROC-AUC: 0.93
McNemar test p-value=0.0000 → Reject H0

Decision Tree -- Accuracy: 0.87, F1: 0.89, ROC-AUC: 0.87
McNemar test p-value=0.0000 → Reject H0

Random Forest -- Accuracy: 0.92, F1: 0.92, ROC-AUC: 0.97
McNemar test p-value=0.0000 → Reject H0
```



##References:
```Zayed, M. E. (2025). Realistic Loan Approval | EDA & Predictions 94 ```
```[Kaggle notebook]. Kaggle. https://www.kaggle.com/code/mohamedzayed2/realistic-loan-approval-eda-predictions-94```