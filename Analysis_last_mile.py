# =====================
# analysis_amazon.py
# =====================
# Dependencies: pip install kagglehub pandas matplotlib
# This file imports Dataset.py (the one you built) and analyzes the data.

# =====================
# Imports & Config
# =====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Import reusable pieces from your Dataset.py
from Dataset import (
    DATASET_ID,
    step1_download,
    step2_locate_tabular_file,
    step3_load_dataframe,
    step7_clean_data,
)

# =====================
# Loaders (modular)
# =====================
def load_raw_dataframe() -> pd.DataFrame:
    data_dir: Path = step1_download(DATASET_ID)
    data_path: Path = step2_locate_tabular_file(data_dir)
    return step3_load_dataframe(data_path)

def load_clean_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    return step7_clean_data(df_raw)

# =====================
# Textual summaries (modular)
# =====================
def print_overview(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== Overview: {name} ===")
    print("Shape:", df.shape)
    print("\nDTypes:")
    print(df.dtypes)
    print("\nHead:")
    print(df.head(5))

def print_numeric_stats(df: pd.DataFrame, name: str) -> None:
    numeric_cols = df.select_dtypes(include="number").columns
    print(f"\n=== Numeric Stats: {name} ===")
    if len(numeric_cols) == 0:
        print("No numeric columns found.")
        return
    print(df[numeric_cols].describe().T)

def show_category_counts(df: pd.DataFrame) -> None:
    candidates = ("Status", "status", "Delivery Status", "delivery_status")
    col = next((c for c in candidates if c in df.columns), None)
    if not col:
        print("\n=== Category Counts ===\nNo status-like column found; skipping.")
        return
    print(f"\n=== Category Counts for '{col}' ===")
    print(df[col].value_counts(dropna=False).head(20))

# =====================
# Plots (modular; each on its own figure)
# =====================
def plot_numeric_histograms(df: pd.DataFrame, cols: list[str], bins: int = 30, tag: str = "") -> None:
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            plt.figure()
            df[c].dropna().plot(kind="hist", bins=bins, title=f"Histogram of {c}{tag}")
            plt.xlabel(c)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

def plot_final_graph(df: pd.DataFrame) -> None:
    """
    Priority A: Bar chart of counts by a status-like column (if present).
    Fallback: Histogram of a likely numeric column.
    """
    # Try to find a categorical status column
    status_candidates = ("Status", "status", "Delivery Status", "delivery_status")
    status_col = next((c for c in status_candidates if c in df.columns), None)

    if status_col:
        counts = df[status_col].value_counts(dropna=False)
        plt.figure()
        counts.plot(kind="bar", title=f"Counts by {status_col}")
        plt.xlabel(status_col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        return

    # Fallback to a likely numeric column
    numeric_candidates = ["distance", "Distance", "delivery_time", "Delivery Time", "cost", "Cost"]
    num_col = next((c for c in numeric_candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])), None)

    if num_col:
        plt.figure()
        df[num_col].dropna().plot(kind="hist", bins=40, title=f"Histogram of {num_col}")
        plt.xlabel(num_col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    else:
        print("\nFinal graph skipped: no suitable status or numeric column found.")

# =====================
# Orchestration (descriptive function calls, not “Step X” names)
# =====================
def main():
    df_raw = load_raw_dataframe()
    df_clean = load_clean_dataframe(df_raw)

    print_overview(df_raw, "RAW")
    print_overview(df_clean, "CLEAN")

    print_numeric_stats(df_raw, "RAW")
    print_numeric_stats(df_clean, "CLEAN")

    show_category_counts(df_raw)
    show_category_counts(df_clean)

    # Optional exploratory histograms
    plot_numeric_histograms(df_raw, ["distance", "delivery_time", "cost"], bins=40, tag=" (RAW)")
    plot_numeric_histograms(df_clean, ["distance", "delivery_time", "cost"], bins=40, tag=" (CLEAN)")

    # Final graph at the end
    plot_final_graph(df_clean)

if __name__ == "__main__":
    main()
