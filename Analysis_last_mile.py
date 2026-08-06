# Analysis_last_mile.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from Dataset import get_train_test_data  # file name: Dataset.py

# -----------------------------------------------------------
# 1. Train the prediction model
# -----------------------------------------------------------
def train_model(area: str | None = "Metropolitian"):

    if area is None:
        X_train, X_test, y_train, y_test, _, _ = get_train_test_data()
        print("\n[train_model] Using ALL areas (no area_filter).")
    else:
        X_train, X_test, y_train, y_test, _, _ = get_train_test_data(
            area_filter=area
        )
        print(f"\n[train_model] Using only Area == '{area}' when possible.")

    # Random Forest model (already reasonably tuned)
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    # 1) Cross-validation on the training set (makes the result more solid)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
    )
    print("\n[train_model] 5-fold CV R² on training data:")
    print(f"  Mean R²:  {cv_scores.mean():.3f}")
    print(f"  Std R²:   {cv_scores.std():.3f}")

    # 2) Fit on all training data
    model.fit(X_train, y_train)

    # 3) Evaluate on test data
    y_pred = model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_pred)
    rf_r2 = r2_score(y_test, y_pred)

    # Naive baseline: always predict mean of y_train
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)

    print(f"\n[train_model] Test MAE (RF):       {rf_mae:.2f} minutes")
    print(f"[train_model] Test R² (RF):        {rf_r2:.3f}")
    print(f"[train_model] Baseline MAE (mean): {baseline_mae:.2f} minutes")
    print(f"[train_model] Baseline R² (mean):  {baseline_r2:.3f}")

    print("\n[train_model] Sample predictions vs actual (first 5):")
    for yp, yt in list(zip(y_pred, y_test))[:5]:
        print(f"  predicted={yp:.1f}  |  actual={yt:.1f}")

    return model, X_test, y_test, baseline_mae, rf_mae, baseline_r2, rf_r2


# -----------------------------------------------------------
# 2. Simulate 50 deliveries and focus on error improvement
# -----------------------------------------------------------
def simulate_50_deliveries(model, X_test, y_test, random_state: int = 0):

    rng = np.random.RandomState(random_state)

    n = len(X_test)
    if n < 50:
        raise ValueError("Not enough test samples to simulate 50 deliveries.")

    sample_indices = rng.choice(n, size=50, replace=False)

    X_sample = X_test.iloc[sample_indices]
    y_true_sample = y_test.iloc[sample_indices]

    # Baseline predictions on these 50 deliveries (use mean of sample as proxy)
    baseline_guess = np.full_like(y_true_sample, fill_value=y_true_sample.mean(), dtype=float)
    baseline_mae_50 = mean_absolute_error(y_true_sample, baseline_guess)

    # Model predictions
    y_pred_sample = model.predict(X_sample)
    rf_mae_50 = mean_absolute_error(y_true_sample, y_pred_sample)

    mae_improvement_50 = (
        (baseline_mae_50 - rf_mae_50) / baseline_mae_50 * 100
        if baseline_mae_50 > 0
        else 0.0
    )

    results = {
        "baseline_mae_50": baseline_mae_50,
        "rf_mae_50": rf_mae_50,
        "mae_improvement_50": mae_improvement_50,
    }

    return results


# -----------------------------------------------------------
# 3. Simple matplotlib dashboard (error-based)
# -----------------------------------------------------------
def plot_dashboard(results):
    """
    Simple dashboard: bar charts for MAE on 50 deliveries.
    """
    labels = ["Baseline MAE", "RF MAE"]
    maes = [results["baseline_mae_50"], results["rf_mae_50"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, maes, color="darkorange")  # <- orange bars
    ax.set_title("Average Absolute Error (50 Deliveries)")
    ax.set_ylabel("Minutes")

    plt.suptitle("Last-Mile Delivery Prediction Prototype")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------
# 4. Main script tying everything together
# -----------------------------------------------------------
def main():
    # 1. Train model on one area (your "city")
    area = "Metropolitian"  # spelling as in CSV after stripping
    model, X_test, y_test, baseline_mae, rf_mae, baseline_r2, rf_r2 = train_model(
        area=area
    )

    # 2. Main KPI: improvement based on MAE reduction (full test set)
    mae_improvement_pct = (baseline_mae - rf_mae) / baseline_mae * 100
    print("\n=== Model Performance Improvement (Prediction Quality) ===")
    print(f"Baseline MAE (test): {baseline_mae:.2f} minutes")
    print(f"RF MAE (test):       {rf_mae:.2f} minutes")
    print(f"MAE improvement:     {mae_improvement_pct:.1f}%")
    print(f"Baseline R²:         {baseline_r2:.3f}")
    print(f"RF R²:               {rf_r2:.3f}")

    # 3. Secondary diagnostic: 50-delivery MAE simulation
    results = simulate_50_deliveries(model, X_test, y_test, random_state=1)

    print("\n=== Simulation Results (50 Deliveries, MAE) ===")
    print(f"Baseline MAE (50):   {results['baseline_mae_50']:.2f} minutes")
    print(f"RF MAE (50):         {results['rf_mae_50']:.2f} minutes")
    print(f"MAE improvement (50): {results['mae_improvement_50']:.1f}%")

    # 4. Show dashboard (error-based)
    plot_dashboard(results)


if __name__ == "__main__":
    main()
