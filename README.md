# CSE 163 Final Project: Using Bayesian Machine Learning to Predict Consumer Purchasing Behavior from Online Retail Transactions

Brandon Hernandez Castellanos & Darren Gao

## 1. Setup

### Requirements

This project uses Python 3 and the packages listed in `requirements.txt`. Install them with:

```
pip install -r requirements.txt
```

If you have multiple Python installations, use `python3 -m pip install -r requirements.txt` instead -- this guarantees the packages install to the same Python that will actually run the scripts.

Installing `pymc` can take a minute or two the first time. That's normal, just let it finish.

### Data

The dataset (`online_retail_II.csv`) is too large for GitHub, so it's included in this repository as `online_retail_II.csv.zip`. **Unzip it before running anything.**

Every script expects the resulting `online_retail_II.csv` to sit in the same folder as the `.py` files. If you place it somewhere else, update the `DATA_PATH` constant near the top of whichever script you're running.

## 2. File descriptions

### Core utilities

- **`eda_core.py`** -- Loads the raw CSV and cleans it: drops cancelled orders, non-positive quantity/price, and non-product stock codes (postage, bank charges, etc.). Every other file imports `load_data` and `clean_data` from here. Running it directly also prints a basic EDA report (row counts, missing data) and saves a monthly revenue plot.
- **`customer_features.py`** -- Computes each customer's Recency, Frequency, Monetary, AvgBasketSize, and Tenure from their *entire* purchase history. Used for RQ2's descriptive analysis (not cutoff-aware, so not used for the predictive models in RQ1/RQ3).
- **`labeling.py`** -- The cutoff-aware version of the feature engineering above. Picks a cutoff date, computes RFM features using only transactions *before* it, and builds both the RQ1 label (did the customer return in the 90 days after the cutoff?) and the RQ3 target (how much did they spend in that window?).

### RQ1 -- Repeat purchase prediction

- **`baseline_model.py`** -- Splits customers into train/test sets, scales the features, and fits the frequentist (scikit-learn) logistic regression baseline.
- **`bayesian_model.py`** -- Fits the Bayesian logistic regression in PyMC on the same features and reports each coefficient's 94% credible interval.
- **`compare_models.py`** -- Runs both RQ1 models above on the same held-out test set and reports accuracy, log-loss, and a calibration comparison. This is the main script for RQ1's final results.

### RQ2 -- Tenure vs. Recency

- **`visualizations.py`** -- Builds four descriptive bar charts (Frequency vs. Monetary, Tenure vs. Monetary, AvgBasketSize distribution, and Tenure vs. Recency) and computes the Pearson correlation between Tenure and Recency, which is RQ2's main result.

### RQ3 -- Predictors of future spend

- **`future_spend_model.py`** -- Fits a Bayesian linear regression on all five customer features against future spend, ranks them by credible-interval distance from zero, and runs a posterior predictive check.

### RQ4 -- Holiday season effect

- **`holiday_model.py`** -- Builds a customer-by-month purchase count table and fits a Bayesian Poisson regression with a holiday-month indicator, directly answering whether the holiday purchase rate is credibly higher.

### EDA / reporting

- **`section6_eda.py`** -- Answers the report's required EDA questions (dataset size, missing data, variables of interest, seven-number summaries) for RQ3/RQ4's feature set.

### Tests

- **`test_eda.py`** -- Unit tests for `report_missing`, `clean_data`, and `compute_customer_features`, using small hand-built DataFrames with known correct answers.

## 3. How to run

All commands below assume you're in a terminal, inside this project folder (the one containing all the `.py` files and the unzipped `online_retail_II.csv`).

Run the tests first, to confirm everything's set up correctly:

```
python3 test_eda.py
```

Then run each research question's script. Each one prints its results directly to the terminal -- there's no separate output file to open.

**EDA report:**
```
python3 section6_eda.py
```

**RQ1 (repeat purchase prediction):**
```
python3 baseline_model.py
python3 bayesian_model.py
python3 compare_models.py
```
`compare_models.py` re-fits both models internally, so it can be run on its own -- the first two are there mainly to see each model's own output in isolation.

**RQ2 (Tenure vs. Recency):**
```
python3 visualizations.py
```

**RQ3 (predictors of future spend):**
```
python3 future_spend_model.py
```

**RQ4 (holiday season effect):**
```
python3 holiday_model.py
```

### A note on runtime

The PyMC scripts (`bayesian_model.py`, `future_spend_model.py`, `holiday_model.py`, and `compare_models.py`) show a live MCMC sampling progress bar and finish in about 2-5 seconds each. That's normal -- not a hang.

## 4. Anything else

- `online_retail_II.csv.zip` must be unzipped before running anything -- see **Setup > Data** above.
- All scripts assume they're run from inside this folder, since they use relative imports between each other (e.g., `labeling.py` imports from `eda_core.py`) and a relative path to the CSV.
