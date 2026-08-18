"""
Tests for eda_core.py and customer_features.py, using small hand-
built DataFrames instead of the full dataset so we know exactly what
the "correct" answer should be.
"""
import pandas as pd
from eda_core import report_missing, clean_data
from customer_features import compute_customer_features


def test_report_missing_counts_correctly():
    df = pd.DataFrame({
        "A": [1, None, 3],
        "B": [None, None, 3],
    })
    missing = report_missing(df)
    assert missing["A"] == 1
    assert missing["B"] == 2


def test_clean_data_removes_cancelled_orders():
    df = pd.DataFrame({
        "Invoice": ["100", "C200"],
        "StockCode": ["85123A", "85123A"],
        "Customer ID": [1.0, 2.0],
        "Quantity": [5, 3],
        "Price": [10.0, 20.0],
        "InvoiceDate": ["2020-01-01", "2020-01-02"],
    })
    cleaned = clean_data(df, verbose=False)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["Invoice"] == "100"


def test_clean_data_keeps_missing_customer_id():
    df = pd.DataFrame({
        "Invoice": ["100", "101"],
        "StockCode": ["85123A", "85123A"],
        "Customer ID": [1.0, None],
        "Quantity": [5, 3],
        "Price": [10.0, 20.0],
        "InvoiceDate": ["2020-01-01", "2020-01-02"],
    })
    cleaned = clean_data(df, verbose=False)
    assert len(cleaned) == 2


def test_compute_customer_features_calculates_correctly():
    df = pd.DataFrame({
        "Invoice": ["100", "101"],
        "Customer ID": [1.0, 1.0],
        "Quantity": [5, 2],
        "Price": [10.0, 5.0],
        "InvoiceDate": pd.to_datetime(["2020-01-01", "2020-01-05"]),
        "Revenue": [50.0, 10.0],
    })
    features = compute_customer_features(df)
    assert len(features) == 1
    assert features.iloc[0]["Frequency"] == 2
    assert features.iloc[0]["Monetary"] == 60.0
    assert features.iloc[0]["Recency"] == 1
    assert features.iloc[0]["Tenure"] == 5
    assert features.iloc[0]["AvgBasketSize"] == 3.5  # (5+2)/2


if __name__ == "__main__":
    test_report_missing_counts_correctly()
    test_clean_data_removes_cancelled_orders()
    test_clean_data_keeps_missing_customer_id()
    test_compute_customer_features_calculates_correctly()
    print("All tests passed!")
