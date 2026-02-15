import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load loan dataset from CSV."""
    return pd.read_csv(path)


def preview_data(df: pd.DataFrame) -> None:
    """Print basic dataset info."""
    print(df.head())
    print(df.info())
    print(df["loan_status"].value_counts())


if __name__ == "__main__":
    path = r"C:\Users\masdr\PycharmProjects\D502 Capstone\Data\Loan_approval_data_2025.csv"
    loan_df = load_data(path)
    preview_data(loan_df)

