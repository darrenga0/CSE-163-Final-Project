"""
Bayesian logistic regression in PyMC for RQ1: same features and
target as baseline_model.py's frequentist model, so the two are
directly comparable.
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from eda_core import load_data, clean_data
from labeling import label_repeat_purchase, DEFAULT_CUTOFF, WINDOW_DAYS
from baseline_model import split_train_test, scale_features


DATA_PATH = "online_retail_II.csv"


def fit_bayesian_logistic(X_train_scaled: pd.DataFrame, y_train: pd.Series,
                          draws: int = 1000, tune: int = 1000,
                          chains: int = 4, random_seed: int = 163):
    """Fit a Bayesian logistic regression with weakly-informative
    Normal(0, 2) priors on the intercept and each coefficient, sampled
    with PyMC's default MCMC sampler (NUTS).

    A Normal(0, 2) prior says: before seeing the data, we think each
    standardized feature's effect is probably small, but we're not
    ruling much out -- it's a mild nudge toward zero, not a strong
    assumption. "Weakly informative" just means the prior is loose
    enough that the data, not our guess, ends up driving the result.

    Returns:
        (model, trace) -- trace holds the posterior samples.
    """
    feature_names = list(X_train_scaled.columns)
    coords = {"feature": feature_names}

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X_data", X_train_scaled.values)
        y_data = pm.Data("y_data", y_train.values)

        intercept = pm.Normal("intercept", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=2, dims="feature")

        logit_p = intercept + pm.math.dot(X_data, beta)
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

        pm.Bernoulli("y_obs", p=p, observed=y_data)

        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          random_seed=random_seed, target_accept=0.9)

    return model, trace


def credible_interval(trace, var_name: str, prob: float = 0.94):
    """94% credible interval computed directly from the raw posterior
    samples via plain numpy -- avoids depending on ArviZ's or
    xarray's summary/quantile APIs, which turned out to vary between
    versions. Returns numpy arrays (or scalars, for a variable with
    no extra dimensions like the intercept).
    """
    alpha = (1 - prob) / 2
    values = trace.posterior[var_name].values  # shape (chain, draw, ...)
    flat = values.reshape(-1, *values.shape[2:])  # (chain*draw, ...)
    low = np.quantile(flat, alpha, axis=0)
    high = np.quantile(flat, 1 - alpha, axis=0)
    return low, high


def check_convergence(trace) -> pd.DataFrame:
    """R-hat sanity check: should be ~1.00 (anything above 1.01 means
    the chains didn't agree with each other and the posterior isn't
    trustworthy yet -- per the proposal's testing plan, this gets
    checked before we trust any credible interval below.
    """
    summary = az.summary(trace, var_names=["intercept", "beta"])
    bad = summary[summary["r_hat"] > 1.01]
    if len(bad) > 0:
        print("WARNING: some parameters have r_hat > 1.01, chains "
              "haven't converged:")
        print(bad)
    else:
        print("Convergence check passed: all r_hat values <= 1.01.")
    return summary


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    clean = clean_data(df, verbose=False)
    labeled = label_repeat_purchase(clean, DEFAULT_CUTOFF, WINDOW_DAYS)

    X_train, X_test, y_train, y_test = split_train_test(labeled)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model, trace = fit_bayesian_logistic(X_train_scaled, y_train)

    print()
    summary = check_convergence(trace)
    print("\nPosterior mean/sd:")
    print(summary[["mean", "sd", "r_hat"]])

    low, high = credible_interval(trace, "beta")
    feature_names = list(X_train_scaled.columns)
    print("\n94% credible intervals per feature:")
    for feature, lo, hi in zip(feature_names, low, high):
        excludes_zero = lo > 0 or hi < 0
        print(f"  {feature}: [{lo:.4f}, {hi:.4f}]"
              f"{'  <- excludes zero' if excludes_zero else ''}")
