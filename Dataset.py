from pathlib import Path
from typing import Optional
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub

# Local dataset location
DATA_DIR = Path("data")
CSV_NAME = "amazon_delivery_dataset.csv"
DATA_PATH = DATA_DIR / CSV_NAME


# -----------------------------------------------------------
# Download dataset using kagglehub
# -----------------------------------------------------------
def download_with_kagglehub() -> Path:
    """
    Download the Amazon delivery dataset using kagglehub and
    copy a CSV file into data/amazon_delivery_dataset.csv.

    Returns:
        Path to the local CSV file (DATA_PATH).
    """
    print("Downloading dataset with kagglehub...")
    download_path = Path(
        kagglehub.dataset_download("sujalsuthar/amazon-delivery-dataset")
    )
    print("Path to dataset files:", download_path)

    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Find a CSV file in the downloaded path
    csv_files = list(download_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {download_path}. "
            "Check the dataset contents."
        )

    # Use the first CSV file found
    source_csv = csv_files[0]
    shutil.copy(source_csv, DATA_PATH)
    print(f"Copied {source_csv.name} to {DATA_PATH}")

    return DATA_PATH


# -----------------------------------------------------------
# Loading and feature engineering
# -----------------------------------------------------------
def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV into a pandas DataFrame.
    If the file doesn't exist, download it with kagglehub first.
    """
    if not path.exists():
        download_with_kagglehub()

    df = pd.read_csv(path)
    return df


def build_features(df: pd.DataFrame):
    """
    Turn the raw DataFrame into:
      - X : feature matrix
      - y : target (delivery time)
      - coords : latitude/longitude columns for routing
    """
    df = df.copy()

    # -------- 1. Target variable ---------------------------------
    target_col = "Delivery_Time"  # adjust if different in your CSV

    if target_col not in df.columns:
        raise KeyError(
            f"Expected target column '{target_col}' not found. "
            "Check the CSV columns and update target_col."
        )

    df = df.dropna(subset=[target_col])

    # -------- 2. Define numeric and categorical columns ----------
    numeric_cols = [
        "Agent_Age",
        "Agent_Rating",
        "Distance",      # example, adjust to actual name
    ]

    categorical_cols = [
        "Weather",
        "Traffic",
        "Vehicle",
        "Area",
        "Category",
    ]

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # -------- 3. Handle missing values ---------------------------
    # Drop rows with missing categorical values
    if categorical_cols:
        df = df.dropna(subset=categorical_cols)

    # Numeric: fill with mean
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # -------- 4. Time-of-day features ----------------------------
    def extract_hour(col_name: str):
        if col_name not in df.columns:
            return None
        col = df[col_name].fillna("12:00").astype(str)
        return col.str.slice(0, 2).astype(int)

    order_hour = extract_hour("Order_Time")
    if order_hour is not None:
        df["Order_Hour"] = order_hour

    pickup_hour = extract_hour("Pickup_Time")
    if pickup_hour is not None:
        df["Pickup_Hour"] = pickup_hour

    hour_cols = [c for c in ["Order_Hour", "Pickup_Hour"] if c in df.columns]

    # -------- 5. Latitude / longitude features -------------------
    latlon_cols = [
        c
        for c in [
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
        ]
        if c in df.columns
    ]

    coords = df[latlon_cols].copy() if latlon_cols else None

    # -------- 6. Final feature matrix X --------------------------
    feature_cols = numeric_cols + hour_cols + categorical_cols + latlon_cols
    if not feature_cols:
        raise ValueError("No feature columns found. Check your column lists.")

    X = df[feature_cols]

    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    y = df[target_col]

    return X, y, coords


# -----------------------------------------------------------
# Train/test split helper
# -----------------------------------------------------------
def get_train_test_data(
    test_size: float = 0.3,  # 70% train / 30% test if you want
    random_state: int = 42,
    area_filter: Optional[str] = None,
):
    """
    Return train/test splits for a given area (e.g. 'Urban').

    Returns:
      X_train, X_test, y_train, y_test, coords_train, coords_test
    """
    df = load_raw_data()
    print(f"\n[get_train_test_data] Raw rows before filtering: {len(df)}")

    # Strip spaces from Area values so "Urban " becomes "Urban"
    if "Area" in df.columns:
        df["Area"] = df["Area"].astype(str).str.strip()

    # Optional: filter by area
    if area_filter is not None and "Area" in df.columns:
        df_area = df[df["Area"] == area_filter].copy()
        print(
            f"[get_train_test_data] Rows after Area == '{area_filter}': {len(df_area)}"
        )

        # If filter removes everything, fall back to using all rows
        if len(df_area) == 0:
            print(
                f"[get_train_test_data] WARNING: no rows found for Area='{area_filter}'. "
                "Falling back to all areas."
            )
        else:
            df = df_area

    # Build features
    X, y, coords = build_features(df)
    print(f"[get_train_test_data] Rows after build_features: {len(X)}")

    if len(X) == 0:
        raise ValueError(
            "[get_train_test_data] No rows left after preprocessing. "
            "Check your filters and dropna logic."
        )

    # Train/test split (with coords if available)
    if coords is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        coords_train = coords_test = None
    else:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            coords_train,
            coords_test,
        ) = train_test_split(
            X, y, coords, test_size=test_size, random_state=random_state
        )

    return X_train, X_test, y_train, y_test, coords_train, coords_test

# -----------------------------------------------------------
if __name__ == "__main__":
    print("[__main__] Running dataset.py for a quick data check")

    # 1. Load raw data (downloads automatically if missing)
    df_raw = load_raw_data()
    print("\n[__main__] Raw data columns:")
    print(df_raw.columns)
    print("\n[__main__] Raw data (first 5 rows):")
    print(df_raw.head())

    # 2. Build features and inspect heads
    X, y, coords = build_features(df_raw)
    print(f"\n[__main__] Feature matrix X shape: {X.shape}")
    print(f"[__main__] Target y shape: {y.shape}")
    print("\n[__main__] Feature matrix X (first 5 rows):")
    print(X.head())
    print("\n[__main__] Target y (first 5 values):")
    print(y.head())

    if coords is not None:
        print(f"\n[__main__] Coords shape: {coords.shape}")
        print("[__main__] Coords (first 5 rows):")
        print(coords.head())

    # 3. Scan for missing values in processed data
    print("\n[__main__] Missing values in processed feature matrix X:")
    print(X.isnull().sum()[X.isnull().sum() > 0])  # only show columns with missing

    if coords is not None:
        print("\n[__main__] Missing values in coords dataframe:")
        print(coords.isnull().sum()[coords.isnull().sum() > 0])

    # 4. Train/test split and inspect training data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        coords_train,
        coords_test,
    ) = get_train_test_data()

    print(f"\n[__main__] X_train shape: {X_train.shape}")
    print(f"[__main__] X_test shape:  {X_test.shape}")
    print(f"[__main__] y_train shape: {y_train.shape}")
    print(f"[__main__] y_test shape:  {y_test.shape}")

    print("\n[__main__] X_train (first 5 rows):")
    print(X_train.head())
    print("\n[__main__] y_train (first 5 values):")
    print(y_train.head())
    print("\nUnique values in Area column (if it exists):")
    if "Area" in df_raw.columns:
        print(df_raw["Area"].unique())