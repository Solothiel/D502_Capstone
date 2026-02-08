import pandas as pd

#loading the dataset
loan_df = pd.read_csv(r"C:\Users\masdr\PycharmProjects\D502 Capstone\Data\Loan_approval_data_2025.csv")

#quick check of data
print(loan_df.head())
print(loan_df.info())

#target distribution check
print(loan_df['loan_status'].value_counts())

