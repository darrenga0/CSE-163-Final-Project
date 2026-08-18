"""
Visualizations for RQ3 and RQ4.

RQ3: Which customer-level features (Recency, Frequency, Monetary,
     AvgBasketSize, Tenure) are the most credible predictors of a
     customer's future spending, based on a Bayesian linear
     regression model?

RQ4: Is there a relationship between how long a customer has been
     active (Tenure) and how recently they've made a purchase
     (Recency), and does this suggest a pattern of customer loyalty
     or customer churn over time?
"""
import pandas as pd
import matplotlib.pyplot as plt
from eda_core import load_data, clean_data
from customer_features import compute_customer_features


DATA_PATH = "online_retail_II.csv"


def plot_frequency_vs_monetary(features: pd.DataFrame, out_path: str) -> dict:
    """Bar chart of average Monetary (spend) across Frequency groups."""
    bins = [0, 1, 3, 10, features["Frequency"].max()]
    labels = ["1 order", "2-3 orders", "4-10 orders", "10+ orders"]
    features = features.copy()
    features["FrequencyGroup"] = pd.cut(features["Frequency"], bins=bins,
                                        labels=labels)
    avg_monetary = features.groupby("FrequencyGroup",
                                    observed=True)["Monetary"].mean()

    plt.figure(figsize=(8, 5))
    avg_monetary.plot(kind="bar", edgecolor="black", color="#4C62B0")
    plt.title("Average Total Spend by Purchase Frequency Group")
    plt.xlabel("Frequency group (number of past orders)")
    plt.ylabel("Average total spend (GBP)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

    return {
        "file": out_path,
        "caption": (
            "Customers who order more frequently spend substantially "
            "more in total, suggesting Frequency is a strong candidate "
            "predictor of spend."
        ),
        "explanation": (
            "This bar chart groups customers into four Frequency "
            "buckets (1 order, 2-3 orders, 4-10 orders, 10+ orders) "
            "and shows each group's average Monetary value. We used "
            "grouped bins rather than a raw scatter plot because "
            "Frequency and Monetary are both heavily right-skewed, "
            "which would otherwise require a log-log scale that's "
            "hard to read at a glance. Readers should take away that "
            "average spend rises sharply and consistently across "
            "Frequency groups, which is early evidence that Frequency "
            "will likely emerge as a credible predictor once the "
            "Bayesian model is fit."
        ),
    }


def plot_tenure_vs_avg_monetary(features: pd.DataFrame, out_path: str) -> dict:
    """Bar chart of average Monetary (spend) across Tenure groups."""
    bins = [0, 180, 365, 545, features["Tenure"].max()]
    labels = ["0-6 mo", "6-12 mo", "12-18 mo", "18+ mo"]
    features = features.copy()
    features["TenureGroup"] = pd.cut(features["Tenure"], bins=bins,
                                     labels=labels)
    avg_monetary = features.groupby("TenureGroup",
                                    observed=True)["Monetary"].mean()

    plt.figure(figsize=(8, 5))
    avg_monetary.plot(kind="bar", edgecolor="black", color="#55A868")
    plt.title("Average Total Spend by Customer Tenure Group")
    plt.xlabel("Tenure group (time since first purchase)")
    plt.ylabel("Average total spend (GBP)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

    return {
        "file": out_path,
        "caption": (
            "Longer tenured customers tend to have higher average "
            "total spend, though the relationship is weaker than for "
            "Frequency."
        ),
        "explanation": (
            "This bar chart groups customers into four Tenure buckets "
            "(time since their first purchase) and shows each group's "
            "average Monetary value. We used a grouped bar chart "
            "rather than a scatter because Tenure is naturally bounded "
            "by the dataset's date range and binning makes the trend "
            "easier to read. Readers should take away that Tenure "
            "shows some positive relationship with spend, but a "
            "shallower one than Frequency -- useful context for "
            "interpreting the Bayesian model's posterior credible "
            "intervals later, since we'd expect Tenure's coefficient "
            "to be positive but with a wider interval."
        ),
    }


def plot_avg_basket_size_distribution(features: pd.DataFrame,
                                      out_path: str) -> dict:
    """Bar chart of customer counts across AvgBasketSize groups."""
    bins = [0, 100, 200, 400, 1000, features["AvgBasketSize"].max()]
    labels = ["1-100", "100-200", "200-400", "400-1000", "1000+"]
    features = features.copy()
    features["BasketGroup"] = pd.cut(features["AvgBasketSize"], bins=bins,
                                     labels=labels)
    counts = features["BasketGroup"].value_counts().reindex(labels)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", edgecolor="black", color="#C44E52")
    plt.title("Number of Customers by Average Basket Size")
    plt.xlabel("Average items per order")
    plt.ylabel("Number of customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

    return {
        "file": out_path,
        "caption": (
            "Most customers order in small, similar quantities per "
            "purchase, with very few placing large bulk orders."
        ),
        "explanation": (
            "This bar chart groups customers into five AvgBasketSize "
            "ranges (average items per order) and counts how many "
            "customers fall into each. We switched from a log-scaled "
            "histogram to grouped bins because AvgBasketSize has a "
            "small number of extreme outliers (one customer averages "
            "87,167 items per order, likely a wholesale account) that "
            "made a log axis hard to read. Readers should take away "
            "that the large majority of customers order 100-400 items "
            "per purchase, and that a handful of extreme bulk buyers "
            "may need separate treatment (e.g. excluding or flagging "
            "wholesale accounts) before AvgBasketSize is used as a "
            "predictor in the Bayesian model."
        ),
    }


def plot_tenure_vs_recency(features: pd.DataFrame, out_path: str) -> dict:
    """Bar chart of average Recency across Tenure groups."""
    bins = [0, 180, 365, 545, features["Tenure"].max()]
    labels = ["0-6 mo", "6-12 mo", "12-18 mo", "18+ mo"]
    features = features.copy()
    features["TenureGroup"] = pd.cut(features["Tenure"], bins=bins,
                                     labels=labels)
    avg_recency = features.groupby("TenureGroup",
                                   observed=True)["Recency"].mean()

    plt.figure(figsize=(8, 5))
    avg_recency.plot(kind="bar", edgecolor="black", color="#8172B2")
    plt.title("Average Recency by Customer Tenure Group")
    plt.xlabel("Tenure group (time since first purchase)")
    plt.ylabel("Average days since last purchase (Recency)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

    return {
        "file": out_path,
        "caption": (
            "Longer-tenured customers tend to have higher average "
            "Recency, suggesting many long-time customers have gone "
            "quiet rather than remaining continuously active."
        ),
        "explanation": (
            "This bar chart groups customers into four Tenure buckets "
            "(time since their first purchase) and shows each group's "
            "average Recency (days since their most recent purchase). "
            "We used the same grouped-bar-chart style as the other "
            "plots for consistency and readability. Readers should "
            "take away that Tenure and Recency move together in this "
            "dataset: customers who have been around the longest also "
            "tend to have the largest gaps since their last purchase, "
            "which reads more as a churn signal than a loyalty signal "
            "-- worth flagging for RQ4, since it suggests tenure alone "
            "isn't a positive sign without also checking Recency."
        ),
    }


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    clean = clean_data(df, verbose=False)
    features = compute_customer_features(clean)

    results = [
        plot_frequency_vs_monetary(features, "frequency_vs_monetary.png"),
        plot_tenure_vs_avg_monetary(features, "tenure_vs_monetary.png"),
        plot_avg_basket_size_distribution(
            features, "basket_size_distribution.png"),
        plot_tenure_vs_recency(features, "tenure_vs_recency.png"),
    ]

    for r in results:
        print(f"\n[{r['file']}]")
        print(f"Caption: {r['caption']}")
        print(f"Explanation: {r['explanation']}")
