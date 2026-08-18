"""
Section 6 (EDA Results) for the Online Retail II dataset.

Directly answers the required EDA questions:
  1. How large is the dataset? What do rows/columns represent?
  2. Does the dataset have missing data?
  3. What are the variables of interest, and why?
  4. Summary of each variable of interest (seven-number summary)

Covers RQ3 and RQ4.
"""
import pandas as pd
from eda_core import load_data, clean_data
from customer_features import compute_customer_features


DATA_PATH = "online_retail_II.csv"


def question_1_dataset_size(df: pd.DataFrame) -> None:
    """Print the dataset's size and describe what rows/columns mean."""
    print("=" * 70)
    print("1. HOW LARGE IS THE DATASET?")
    print("=" * 70)
    rows, cols = df.shape
    print(f"Rows: {rows:,}")
    print(f"Columns: {cols}")
    print(f"Column names: {list(df.columns)}")
    print(
        "\nEach row represents a single line item within a transaction -- "
        "one product, at one quantity, on one invoice.\n"
        "Columns capture: the invoice/product identifiers (Invoice, "
        "StockCode, Description), the sale details (Quantity, Price, "
        "InvoiceDate), and the customer (Customer ID, Country).\n"
    )


def question_2_missing_data(df: pd.DataFrame) -> pd.Series:
    """Report missing-data counts/percentages and explain the plan."""
    print("=" * 70)
    print("2. DOES THE DATASET HAVE MISSING DATA?")
    print("=" * 70)
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct.round(2),
    })
    has_missing = summary[summary["missing_count"] > 0]
    if has_missing.empty:
        print("No missing data found (verified via df.isna().sum()).")
    else:
        print("Columns with missing data (via df.isna().sum()):")
        print(has_missing)
        print(
            "\nPlan: Description missingness (0.41%) is minor and unused "
            "in this analysis. Customer ID missingness (22.77%) matters "
            "for customer-level analysis, so those rows are kept for "
            "transaction-level analysis but dropped specifically when "
            "computing per-customer features, where a known customer is "
            "required.\n"
        )
    return missing_count


def question_3_variables_of_interest() -> None:
    """Print the variables of interest for RQ3 and RQ4, and why."""
    print("=" * 70)
    print("3. VARIABLES OF INTEREST")
    print("=" * 70)
    print(
        "RQ3: Which customer-level features (Recency, Frequency, "
        "Monetary, AvgBasketSize, Tenure) are the most credible "
        "predictors of a customer's future spending, based on a "
        "Bayesian linear regression model?\n"
        "\n"
        "RQ4: Is there a relationship between how long a customer has "
        "been active (Tenure) and how recently they've made a "
        "purchase (Recency), and does this suggest a pattern of "
        "customer loyalty or customer churn over time?\n"
        "\n"
        "Variables of interest (engineered from Invoice, InvoiceDate, "
        "Quantity, Price, and Customer ID):\n"
        "  - Recency:      days since a customer's last purchase\n"
        "  - Frequency:    number of distinct orders placed\n"
        "  - Monetary:     total amount spent\n"
        "  - AvgBasketSize: average number of items per order\n"
        "  - Tenure:       days since a customer's first purchase\n"
        "\n"
        "These five are used because they are the direct inputs named "
        "in RQ3's Bayesian linear regression model, and Tenure and "
        "Recency specifically are the two variables RQ4 compares "
        "directly.\n"
    )


def question_4_variable_summaries(features: pd.DataFrame) -> pd.DataFrame:
    """Print/return the seven-number summary for the five variables."""
    print("=" * 70)
    print("4. SUMMARY OF EACH VARIABLE OF INTEREST")
    print("=" * 70)
    cols = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "Tenure"]
    summary = features[cols].describe()
    print(summary)
    print(
        "\nFrequency, Monetary, and AvgBasketSize are all heavily "
        "right-skewed: a small number of high-activity customers pull "
        "the mean well above the median. Recency and Tenure are less "
        "skewed, since both are naturally bounded by the dataset's "
        "date range.\n"
    )
    return summary


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    question_1_dataset_size(df)
    question_2_missing_data(df)
    question_3_variables_of_interest()

    clean = clean_data(df, verbose=False)
    features = compute_customer_features(clean)
    question_4_variable_summaries(features)
