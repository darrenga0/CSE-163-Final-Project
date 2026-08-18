"""
RQ1 step 4: compare the frequentist and Bayesian logistic regression
models on the held-out test set the two were never fit on -- test
accuracy, log-loss, and a calibration check, per the proposal's
Method section for RQ1.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from eda_core import load_data, clean_data
from labeling import label_repeat_purchase, DEFAULT_CUTOFF, WINDOW_DAYS
from baseline_model import (FEATURES, split_train_test, scale_features,
                            fit_baseline_logistic)
from bayesian_model import fit_bayesian_logistic, credible_interval


DATA_PATH = "online_retail_II.csv"


def bayesian_predict_proba(trace, X_scaled: pd.DataFrame) -> np.ndarray:
    """Predicted probabilities from the Bayesian model, using the
    posterior mean of the intercept and each coefficient as a point
    estimate.

    Simplification made for time: averaging the prediction over every
    individual posterior sample (rather than just the posterior mean)
    would be more fully Bayesian and would naturally widen predicted
    probabilities for uncertain cases. Using the posterior mean is a
    standard, defensible simplification here, and keeps this
    apples-to-apples with the frequentist model, which is also just a
    single point-estimate coefficient per feature.
    """
    intercept_mean = trace.posterior["intercept"].values.mean()
    beta_mean = (trace.posterior["beta"].values
                 .reshape(-1, X_scaled.shape[1]).mean(axis=0))

    logits = intercept_mean + X_scaled.values @ beta_mean
    probs = 1 / (1 + np.exp(-logits))
    return probs


def calibration_table(y_true: pd.Series, probs: np.ndarray,
                      bins: int = 5) -> pd.DataFrame:
    """Bucket test customers into `bins` groups by predicted
    probability, and compare each bucket's average predicted
    probability to its actual observed repeat-purchase rate. A
    well-calibrated model has these two columns sitting close
    together in every row.
    """
    table = pd.DataFrame({"y_true": y_true.values, "pred_prob": probs})
    table["bucket"] = pd.qcut(table["pred_prob"], q=bins, duplicates="drop")
    summary = table.groupby("bucket", observed=True).agg(
        avg_predicted=("pred_prob", "mean"),
        observed_rate=("y_true", "mean"),
        n=("y_true", "size"),
    )
    return summary


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    clean = clean_data(df, verbose=False)
    labeled = label_repeat_purchase(clean, DEFAULT_CUTOFF, WINDOW_DAYS)

    X_train, X_test, y_train, y_test = split_train_test(labeled)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Frequentist baseline
    freq_model = fit_baseline_logistic(X_train_scaled, y_train)
    freq_probs = freq_model.predict_proba(X_test_scaled)[:, 1]
    freq_preds = freq_model.predict(X_test_scaled)

    # Bayesian model (re-fit here so this script can run standalone)
    print("Fitting Bayesian model...")
    bayes_model, trace = fit_bayesian_logistic(X_train_scaled, y_train)
    bayes_probs = bayesian_predict_proba(trace, X_test_scaled)
    bayes_preds = (bayes_probs >= 0.5).astype(int)

    print("\n" + "=" * 50)
    print("TEST-SET COMPARISON")
    print("=" * 50)
    print(f"{'Metric':<20}{'Frequentist':>15}{'Bayesian':>15}")
    print(f"{'Accuracy':<20}"
          f"{accuracy_score(y_test, freq_preds):>15.3f}"
          f"{accuracy_score(y_test, bayes_preds):>15.3f}")
    print(f"{'Log-loss':<20}"
          f"{log_loss(y_test, freq_probs):>15.3f}"
          f"{log_loss(y_test, bayes_probs):>15.3f}")

    print("\nFrequentist calibration (predicted vs. actual repeat rate):")
    print(calibration_table(y_test, freq_probs))

    print("\nBayesian calibration (predicted vs. actual repeat rate):")
    print(calibration_table(y_test, bayes_probs))

    low, high = credible_interval(trace, "beta")
    print("\nBayesian 94% credible intervals (for reference):")
    for feature, lo, hi in zip(FEATURES, low, high):
        excludes_zero = lo > 0 or hi < 0
        print(f"  {feature}: [{lo:.4f}, {hi:.4f}]"
              f"{'  <- excludes zero' if excludes_zero else ''}")
