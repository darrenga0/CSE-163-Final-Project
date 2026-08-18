"""
Train/test split and (next) baseline logistic regression for RQ1:
predicting repeat purchase within 90 days from RFM features.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from eda_core import load_data, clean_data
from labeling import label_repeat_purchase, DEFAULT_CUTOFF, WINDOW_DAYS


DATA_PATH = "online_retail_II.csv"
FEATURES = ["Recency", "Frequency", "Monetary"]
TARGET = "Repeat"


def split_train_test(labeled: pd.DataFrame, test_size: float = 0.2,
                     random_state: int = 163):
    """Split the labeled customer table into train/test sets on
    [Recency, Frequency, Monetary] -> Repeat.

    Stratified on the label so both sets keep roughly the same
    repeat-purchase rate -- otherwise a random split could get
    unlucky and put too few (or too many) repeat customers in test.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = labeled[FEATURES]
    y = labeled[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Standardize features to mean 0, std 1 (fit on train only, then
    applied to both).

    Necessary here because Recency/Frequency/Monetary are on wildly
    different scales (Monetary ranges into the hundreds of thousands
    of pounds, Frequency only into the hundreds of orders). Without
    scaling, a coefficient near zero could just mean "this feature's
    raw units are huge," not "this feature doesn't matter" -- and
    that would make the coefficients impossible to compare fairly,
    which is exactly what RQ1 needs to do.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler


def fit_baseline_logistic(X_train: pd.DataFrame,
                          y_train: pd.Series) -> LogisticRegression:
    """Fit a plain (frequentist) logistic regression on the training
    set -- the baseline RQ1's Bayesian model gets compared against.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    clean = clean_data(df, verbose=False)
    labeled = label_repeat_purchase(clean, DEFAULT_CUTOFF, WINDOW_DAYS)

    X_train, X_test, y_train, y_test = split_train_test(labeled)

    print(f"Total customers: {len(labeled):,}")
    print(
        f"Train: {len(X_train):,} customers({y_train.mean():.1%} repeat rate)")
    print(
        f"Test:  {len(X_test):,} customers ({y_test.mean():.1%} repeat rate)")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = fit_baseline_logistic(X_train_scaled, y_train)
    print("\nBaseline logistic regression coefficients (on scaled features):")
    for feature, coef in zip(FEATURES, model.coef_[0]):
        print(f"  {feature}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")
    train_acc = model.score(X_train_scaled, y_train)
    print(f"\nTraining accuracy (sanity check only, real evaluation is "
          f"step 4): {train_acc:.1%}")
