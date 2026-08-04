"""
Core data loading and cleaning utilities for the Online Retail II
dataset. This is the foundational module; customer_features.py,
visualizations.py, section6_eda.py, and test_eda.py all import from
this file.
"""
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "online_retail_II.csv"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw sales data from a CSV file."""
    df = pd.read_csv(path, encoding="ISO-8859-1")
    return df


def report_shape(df: pd.DataFrame) -> None:
    """Print the number of rows and columns in the dataset."""
    rows, cols = df.shape
    print(f"Dataset has {rows} rows and {cols} columns.")
    print(f"Columns: {list(df.columns)}")


def report_missing(df: pd.DataFrame) -> pd.Series:
    """Return count and percent of missing values per column."""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct.round(2),
    })
    print(summary[summary["missing_count"] > 0])
    return missing_count


NON_PRODUCT_CODES = ["POST", "DOT", "M", "m", "B", "BANK CHARGES", "D", "C2"]


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    General purpose cleaning for transaction level analysis (does NOT
    require a Customer ID, since monthly revenue doesn't care who
    made the purchase):
      - Removed cancelled orders (InvoiceNo starting with 'C')
      - Removed non-positive quantities/prices (this also removes the
        "Adjust bad debt" rows, since those carry negative Price)
      - Removed non-product StockCodes (postage, manual entries, bank
        charges, etc.) these aren't customer purchases and would
        distort revenue totals
    Rows with a missing Customer ID are kept here, since they're still
    valid sales; They're only excluded later, at the feature,
    engineering step, where knowing which customer made the
    purchase is required.
    """
    step_counts = [("Raw data", len(df))]

    clean = df[~df["Invoice"].astype(str).str.startswith("C")]
    step_counts.append(("After removing cancelled orders", len(clean)))

    clean = clean[(clean["Quantity"] > 0) & (clean["Price"] > 0)]
    step_counts.append(("After removing non-positive qty/price", len(clean)))

    clean = clean[~clean["StockCode"].astype(str).isin(NON_PRODUCT_CODES)]
    step_counts.append(("After removing non-product codes", len(clean)))

    clean = clean.copy()
    clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"])
    clean["Revenue"] = clean["Quantity"] * clean["Price"]

    if verbose:
        print("\nRow counts through cleaning pipeline:")
        for label, count in step_counts:
            print(f"  {label}: {count:,} rows")

    return clean


def summarize_quantitative(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Seven-number summary for quantitative variables of interest."""
    return df[columns].describe()


def plot_monthly_revenue(df: pd.DataFrame, out_path: str) -> None:
    """Line plot of total revenue by month -- shows trend/seasonality."""
    monthly = df.set_index("InvoiceDate").resample("ME")["Revenue"].sum()

    plt.figure(figsize=(10, 5))
    monthly.plot(marker="o")
    plt.title("Total Monthly Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue (GBP)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    report_shape(df)
    report_missing(df)
    clean = clean_data(df)
    print(summarize_quantitative(clean, ["Quantity", "Price", "Revenue"]))
    plot_monthly_revenue(clean, "monthly_revenue.png")
