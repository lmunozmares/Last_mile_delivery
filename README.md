# Last-Mile Delivery Time Prediction

Predicting delivery duration from conditions known at dispatch, using Amazon
logistics data. A Random Forest halves the average prediction error versus
quoting the fleet average.

## Problem

Accurate delivery estimates drive customer expectations, courier scheduling and
route planning. The question: how much of delivery duration is explained by
conditions observable at dispatch, and can a model meaningfully beat quoting an
average?

## Data

[Amazon Delivery Dataset](https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset)
(Kaggle), downloaded programmatically via `kagglehub`. 43,739 deliveries,
filtered to metropolitan areas (32,634 after preprocessing).

Features: agent age and rating, weather, traffic, vehicle type, area, product
category, order and pickup hour, store and drop coordinates.

## Approach

| Model | Role |
|---|---|
| Mean prediction | Baseline. Quotes the training-set average for every delivery. |
| Random Forest | 400 trees, max depth 15, min 3 samples per leaf. |

Categorical features one-hot encoded, hour extracted from order and pickup
timestamps. 70/30 train/test split, with 5-fold cross-validation on the
training set.

## Results

| Model | MAE (minutes) | R² |
|---|---|---|
| Mean baseline | 41.26 | -0.000 |
| **Random Forest** | **20.63** | **0.729** |

**5-fold CV R² on training data: 0.721 (± 0.011)**

The model halves average error, from 41 minutes to 21. The baseline R² of zero
is definitional: predicting the mean explains none of the variance.

## Discussion

### The model generalises

Cross-validated R² (0.721) and held-out test R² (0.729) are effectively
identical, and the standard deviation across folds is 0.011. A Random Forest
with 400 trees at depth 15 has ample capacity to memorise, so the absence of a
train/test gap is the result worth trusting here. Nothing suggests overfitting.

### Why R² is high here, and low in time-series work

For contrast, a
[household electricity forecasting project](https://github.com/lmunozmares/Household_Electricity_Demand_Forecasting)
using the same toolkit reaches R² of only 0.30. The difference is the problem,
not the model.

This is cross-sectional prediction with genuinely informative features: traffic
conditions, weather, agent rating and vehicle type are causally related to how
long a delivery takes, and they are known before it starts. Time-series
forecasting of a single household's consumption has no comparable predictors,
only the series' own history, and the signal is swamped by discrete appliance
events.

The lesson is that R² is a property of the problem as much as the model. Judging
a forecasting model against a regression model's R² would be a category error.

### 20 minutes is an improvement, not a solution

Halving error is a real gain, but a 21-minute average error is still wide for a
customer-facing ETA. The sample predictions show individual misses of 30 to 35
minutes. Before this could support a delivery promise, error would need to be
characterised by segment rather than reported as a single average: which
conditions produce the large misses, and are they predictable in advance?

### Scope

Restricted to metropolitan areas, which is 75% of the dataset. Rural and
semi-urban deliveries have different distance and infrastructure profiles, and
the model has not been validated on them.

## Structure

- `Dataset.py` — download, cleaning, feature engineering, train/test split
- `Analysis_last_mile.py` — training, evaluation, comparison, dashboard

## Running it

```bash
pip install -r requirements.txt
python Analysis_last_mile.py
```

The dataset downloads automatically on first run (requires Kaggle API
credentials at `~/.kaggle/kaggle.json`).

## Known limitations

- Missing numeric values are imputed with the column mean **before** the
  train/test split, leaking a small amount of test information. Should sit
  inside a pipeline fitted on training data only.
- Hyperparameters were set by hand rather than searched.
- Coordinates are used as raw latitude/longitude. Deriving distance and local
  density is likely the largest untapped signal.
- `simulate_50_deliveries` uses the sampled test mean as its baseline, while the
  main evaluation correctly uses the training mean. Inconsistent, though it
  makes the reported improvement conservative rather than flattering.

---

**Luis Muñoz Mares** — MSc Managing with Data & AI, Grenoble Ecole de Management
[LinkedIn](https://linkedin.com/in/luismunozmares)
