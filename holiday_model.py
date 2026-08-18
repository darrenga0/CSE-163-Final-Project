"""
RQ4: is purchase frequency credibly higher in Nov-Dec than the rest
of the year? Bayesian Poisson regression in PyMC.

Simplified from the original plan of "two separate Poisson models,
one per group, then compute the difference of their posteriors" to
one Poisson regression with a holiday indicator as a predictor. Both
approaches answer the same question, but this way there's only one
model to fit and one coefficient to interpret: since a Poisson model
uses a log link, exp(beta_holiday) IS the holiday-vs-non-holiday rate
ratio, so beta_holiday's credible interval and P(beta_holiday > 0)
already are the group comparison -- no extra step needed.
"""
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

from eda_core import load_data, clean_data


DATA_PATH = "online_retail_II.csv"
HOLIDAY_MONTHS = [11, 12]


def monthly_purchase_counts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (customer, active month): how many distinct
    invoices they placed that month, and whether that month is a
    holiday month (Nov-Dec).
    """
    df = df.dropna(subset=["Customer ID"]).copy()
    df["Month"] = df["InvoiceDate"].dt.to_period("M")

    counts = (df.groupby(["Customer ID", "Month"])["Invoice"]
              .nunique().reset_index(name="PurchaseCount"))
    counts["IsHoliday"] = (counts["Month"].dt.month.isin(HOLIDAY_MONTHS)
                           .astype(int))
    return counts


def fit_bayesian_poisson(counts: pd.DataFrame, draws: int = 1000,
                         tune: int = 1000, chains: int = 4,
                         random_seed: int = 163):

    """Poisson regression: PurchaseCount ~ Poisson(mu),
    log(mu) = intercept + beta_holiday * IsHoliday.

    Weakly-informative Normal(0, 1) priors here -- tighter than the
    Normal(0, 2) used for RQ1/RQ3, since this is on the log scale
    where even a coefficient of 1 already means "the rate multiplies
    by e ≈ 2.7", so a wide prior would be putting real weight on
    implausibly large swings.
    """
    with pm.Model() as model:
        is_holiday = pm.Data("is_holiday", counts["IsHoliday"].values)
        y_data = pm.Data("y_data", counts["PurchaseCount"].values)

        intercept = pm.Normal("intercept", mu=0, sigma=1)
        beta_holiday = pm.Normal("beta_holiday", mu=0, sigma=1)

        log_mu = intercept + beta_holiday * is_holiday
        mu = pm.math.exp(log_mu)

        pm.Poisson("y_obs", mu=mu, observed=y_data)

        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          random_seed=random_seed, target_accept=0.9)

    return model, trace


def summarize_holiday_effect(trace) -> None:
    """Report the rate ratio, its 94% credible interval, and the
    probability the holiday rate is actually higher -- the direct
    answer to RQ4.

    Credible interval computed directly from posterior samples (not
    ArviZ's summary() table), since its interval column naming varies
    between ArviZ versions.
    """
    summary = az.summary(trace, var_names=["intercept", "beta_holiday"])
    print(summary[["mean", "sd", "r_hat"]])

    alpha = (1 - 0.94) / 2
    beta_samples = trace.posterior["beta_holiday"].values.flatten()
    low = float(np.quantile(beta_samples, alpha))
    high = float(np.quantile(beta_samples, 1 - alpha))
    print(f"\nbeta_holiday 94% credible interval: [{low:.4f}, {high:.4f}]"
          f"{'  <- excludes zero' if (low > 0 or high < 0) else ''}")

    prob_higher = (beta_samples > 0).mean()

    non_holiday_rate = np.exp(trace.posterior["intercept"].values.mean())
    holiday_rate = np.exp(trace.posterior["intercept"].values.mean()
                          + trace.posterior["beta_holiday"].values.mean())

    print(f"\nEstimated non-holiday monthly rate: {non_holiday_rate:.3f} "
          f"purchases/customer")
    print(f"Estimated holiday monthly rate:     {holiday_rate:.3f} "
          f"purchases/customer")
    print(f"P(holiday rate > non-holiday rate): {prob_higher:.1%}")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    clean = clean_data(df, verbose=False)
    counts = monthly_purchase_counts(clean)

    print(f"Customer-months: {len(counts):,}")
    print(f"Mean count: {counts['PurchaseCount'].mean():.3f}, "
          f"variance: {counts['PurchaseCount'].var():.3f} "
          f"(close together -> plain Poisson is a reasonable fit)")

    model, trace = fit_bayesian_poisson(counts)

    print()
    summarize_holiday_effect(trace)
