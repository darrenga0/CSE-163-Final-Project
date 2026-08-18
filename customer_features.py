"""
Engineer the full set of customer-level features needed for RQ3 and
RQ4: Recency, Frequency, Monetary, AvgBasketSize, and Tenure.
"""
import pandas as pd
from eda_core import load_data, clean_data


DATA_PATH = "online_retail_II.csv"


def compute_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary, AvgBasketSize, and Tenure
    per customer from cleaned transaction data.

    Args:
        df: cleaned transaction-level DataFrame (output of
            eda_core.clean_data), with an 'InvoiceDate' column
            already converted to datetime and a 'Revenue' column.

    Returns:
        One row per customer with columns: Customer ID, Recency,
        Frequency, Monetary, Tenure, AvgBasketSize.
    """
    before = len(df)
    df = df.dropna(subset=["Customer ID"])
    print(f"Dropped {before - len(df):,} rows with missing Customer ID "
          f"({(before - len(df)) / before:.1%} of input).")

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    features = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        TotalItems=("Quantity", "sum"),
        Tenure=("InvoiceDate", lambda x: (snapshot_date - x.min()).days),
    ).reset_index()

    features["AvgBasketSize"] = features["TotalItems"] / features["Frequency"]
    features = features.drop(columns=["TotalItems"])

    return features


def summarize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Seven-number summary for all five predictor variables."""
    cols = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "Tenure"]
    return features[cols].describe()


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    clean = clean_data(df, verbose=False)
    features = compute_customer_features(clean)
    print(f"\nFeature table has {features.shape[0]} customers and "
          f"{features.shape[1]} columns.")
    print(summarize_features(features))
    features.to_csv("customer_features.csv", index=False)
    print("\nSaved customer features to customer_features.csv")
