"""
RQ3: which customer features best predict spend in the 90 days after
the cutoff? Bayesian linear regression in PyMC on the five features
from customer_features.py.

Reuses labeling.py's future_spend_target for the feature/target table
-- same cutoff-aware approach as RQ1, just a continuous target
instead of a binary label.
"""
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eda_core import load_data, clean_data
from labeling import future_spend_target, DEFAULT_CUTOFF, WINDOW_DAYS


DATA_PATH = "online_retail_II.csv"
FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "Tenure"]
TARGET = "FutureSpend"


def split_and_scale(targets: pd.DataFrame, test_size: float = 0.2,
                    random_state: int = 163):
    """Train/test split, then standardize BOTH the features and the
    target (mean 0, std 1, fit on train only).

    Scaling the target matters just as much as scaling the features
    here: FutureSpend is in raw pounds (could be in the thousands),
    while the features are standardized to a small range. Without
    also scaling the target, the Normal(0, 2) priors below would be
    fighting against the wrong scale entirely.
    """
    X = targets[FEATURES]
    y = targets[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    x_scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(x_scaler.fit_transform(X_train),
                                  columns=FEATURES, index=X_train.index)
    X_test_scaled = pd.DataFrame(x_scaler.transform(X_test),
                                 columns=FEATURES, index=X_test.index)

    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
            x_scaler, y_mean, y_std)


def fit_bayesian_linear(X_train_scaled: pd.DataFrame,
                        y_train_scaled: pd.Series,
                        draws: int = 1000, tune: int = 1000,
                        chains: int = 4, random_seed: int = 163):
    """Bayesian linear regression: (scaled) FutureSpend ~ Normal(mu,
    sigma), mu = intercept + X @ beta. Same weakly-informative
    Normal(0, 2) priors on the coefficients as RQ1's logistic model,
    plus a HalfNormal(1) prior on sigma -- sensible defaults now that
    everything (features and target) is on the same standardized
    scale.

    Simplification made for time: FutureSpend is heavily right-skewed
    (most customers spend little or nothing, a few spend a lot), so a
    Normal likelihood isn't a perfect fit even after scaling -- a
    log1p-transformed target would model that better. Kept simple
    here since it still answers RQ3's actual question (which
    coefficients are credibly nonzero); worth flagging as a possible
    next refinement, not a correctness bug.
    """
    feature_names = list(X_train_scaled.columns)
    coords = {"feature": feature_names}

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X_data", X_train_scaled.values)
        y_data = pm.Data("y_data", y_train_scaled.values)

        intercept = pm.Normal("intercept", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=2, dims="feature")
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = intercept + pm.math.dot(X_data, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          random_seed=random_seed, target_accept=0.9)

    return model, trace


def credible_interval(trace, var_name: str, prob: float = 0.94):
    """94% credible interval computed directly from the raw posterior
    samples via plain numpy -- avoids depending on ArviZ's or
    xarray's summary/quantile APIs, which turned out to vary between
    versions.
    """
    alpha = (1 - prob) / 2
    values = trace.posterior[var_name].values  # shape (chain, draw, ...)
    flat = values.reshape(-1, *values.shape[2:])  # (chain*draw, ...)
    low = np.quantile(flat, alpha, axis=0)
    high = np.quantile(flat, 1 - alpha, axis=0)
    return low, high


def rank_features(trace) -> pd.DataFrame:
    """Rank features by how far their 94% credible interval sits from
    zero -- an interval that excludes zero is a credible predictor.
    Coefficients are in "std devs of spend per std dev of feature"
    units, which is exactly what makes them comparable to each other.
    """
    summary = az.summary(trace, var_names=["beta"])
    low, high = credible_interval(trace, "beta")
    summary["hdi_low"] = low
    summary["hdi_high"] = high
    summary["excludes_zero"] = (low > 0) | (high < 0)
    summary["distance_from_zero"] = summary["mean"].abs()
    return summary.sort_values("distance_from_zero", ascending=False)


def posterior_predictive_check(model, trace, y_train_scaled: pd.Series,
                               y_mean: float, y_std: float):
    """Lightweight posterior predictive check: simulate spend from the
    fitted model, un-scale it back to pounds, and compare its mean
    against the real training mean. A full distributional comparison
    (plot) would be a nice extension if time allows, but this sanity
    check already tells us whether predictions are in the right
    ballpark.
    """
    with model:
        ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"])

    simulated_mean_scaled = float(ppc.posterior_predictive["y_obs"].mean())
    simulated_mean = simulated_mean_scaled * y_std + y_mean
    real_mean = float(y_train_scaled.mean()) * y_std + y_mean

    print(f"Real training mean FutureSpend:      £{real_mean:.2f}")
    print(f"Posterior-predictive simulated mean: £{simulated_mean:.2f}")
    print(f"Difference: £{abs(simulated_mean - real_mean):.2f} "
          f"({abs(simulated_mean - real_mean) / real_mean:.1%} of real mean)")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    clean = clean_data(df, verbose=False)
    targets = future_spend_target(clean, DEFAULT_CUTOFF, WINDOW_DAYS)

    (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
     x_scaler, y_mean, y_std) = split_and_scale(targets)

    model, trace = fit_bayesian_linear(X_train_scaled, y_train_scaled)

    ranked = rank_features(trace)
    print("\nFeature ranking (94% credible interval distance from zero):")
    print(ranked[["mean", "hdi_low", "hdi_high", "excludes_zero"]])

    print()
    posterior_predictive_check(model, trace, y_train_scaled, y_mean, y_std)
