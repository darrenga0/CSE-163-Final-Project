"""
Cutoff-aware customer features and future-window labeling.

Builds the RQ1 label (did a customer return within 90 days of the
cutoff?) and the RQ3 target (how much did they spend in that window?)
off the same shared pieces, as scoped in the handoff doc.

The core idea: pick a cutoff date that splits the cleaned transaction
history into a "past" (used to compute a customer's Recency,
Frequency, Monetary, etc. *as of* the cutoff) and a "future" window
(what actually happened in the window_days after it, which becomes
the label/target). Because the dataset only runs through 2011-12-09,
the cutoff has to leave at least window_days of transactions after
it, or there's no way to know the true label for customers near the
end of the dataset.
"""
import pandas as pd
from eda_core import load_data, clean_data


DATA_PATH = "online_retail_II.csv"
WINDOW_DAYS = 90

# Dataset runs 2009-12-01 to 2011-12-09 (~738 days total). This cutoff
# leaves 648 days of history to compute RFM from, and ~90 days of
# transactions after it to check the label against.
DEFAULT_CUTOFF = pd.Timestamp("2011-09-10")


def compute_customer_features_cutoff(
        df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Cutoff-aware version of customer_features.compute_customer_features:
    Recency, Frequency, Monetary, AvgBasketSize, and Tenure computed
    from only the transactions strictly before cutoff_date, and
    measured relative to cutoff_date instead of the dataset's max
    date.

    Args:
        df: cleaned transaction-level DataFrame (output of
            eda_core.clean_data).
        cutoff_date: the "as of" date features are computed relative
            to. Only transactions with InvoiceDate < cutoff_date are
            used.

    Returns:
        One row per customer with at least one transaction before
        cutoff_date. Columns: Customer ID, Recency, Frequency,
        Monetary, Tenure, AvgBasketSize.
    """
    past = df[df["InvoiceDate"] < cutoff_date]
    past = past.dropna(subset=["Customer ID"])

    features = past.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (cutoff_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        TotalItems=("Quantity", "sum"),
        Tenure=("InvoiceDate", lambda x: (cutoff_date - x.min()).days),
    ).reset_index()

    features["AvgBasketSize"] = features["TotalItems"] / features["Frequency"]
    features = features.drop(columns=["TotalItems"])

    return features


def get_future_window(df: pd.DataFrame, cutoff_date: pd.Timestamp,
                      window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """Transactions in the window_days immediately after cutoff_date.

    Shared by both targets below: RQ1 only cares whether a customer
    shows up at all in this window, RQ3 cares how much they spent.
    """
    window_end = cutoff_date + pd.Timedelta(days=window_days)
    future = df[(df["InvoiceDate"] >= cutoff_date) &
                (df["InvoiceDate"] < window_end)]
    return future.dropna(subset=["Customer ID"])


def label_repeat_purchase(df: pd.DataFrame, cutoff_date: pd.Timestamp,
                          window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """RQ1 label: for every customer active before cutoff_date, 1 if
    they have any invoice in the window_days after cutoff_date, else
    0.
    """
    features = compute_customer_features_cutoff(df, cutoff_date)
    future = get_future_window(df, cutoff_date, window_days)

    returning_ids = set(future["Customer ID"].unique())
    features["Repeat"] = (features["Customer ID"].isin(
        returning_ids).astype(int))

    return features


def future_spend_target(df: pd.DataFrame, cutoff_date: pd.Timestamp,
                        window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """RQ3 target: for every customer active before cutoff_date, total
    Revenue spent in the window_days after cutoff_date (0 if they
    didn't return at all).
    """
    features = compute_customer_features_cutoff(df, cutoff_date)
    future = get_future_window(df, cutoff_date, window_days)

    future_spend = future.groupby("Customer ID")["Revenue"].sum()
    features["FutureSpend"] = (features["Customer ID"].map(
        future_spend).fillna(0))

    return features


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    clean = clean_data(df, verbose=False)

    labeled = label_repeat_purchase(clean, DEFAULT_CUTOFF)
    print(f"Cutoff date: {DEFAULT_CUTOFF.date()}")
    print(f"Customers active before cutoff: {len(labeled):,}")
    print(f"Repeat purchase rate: {labeled['Repeat'].mean():.1%}")
    print(labeled["Repeat"].value_counts().rename("count"))

    targets = future_spend_target(clean, DEFAULT_CUTOFF)
    print(f"\nMedian future spend (returning customers only): "
          f"{targets.loc[targets['FutureSpend'] > 0,
                         'FutureSpend'].median():.2f}")
