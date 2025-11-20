# =====================
# Imports & Global Config
# =====================
from pathlib import Path
import pandas as pd
import kagglehub

DATASET_ID = "sujalsuthar/amazon-delivery-dataset"

# =====================
# Step 1 — Download dataset (cached)
# =====================
def step1_download(dataset_id: str) -> Path:
    data_dir = Path(kagglehub.dataset_download(dataset_id))
    print("Dataset folder:", data_dir)
    return data_dir

# =====================
# Step 2 — Locate a tabular file (CSV preferred, else Parquet)
# =====================
def step2_locate_tabular_file(data_dir: Path) -> Path:
    candidates = list(data_dir.rglob("*.csv")) + list(data_dir.rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No CSV/Parquet files found in {data_dir}")
    data_path = next((p for p in candidates if p.suffix == ".csv"), candidates[0])
    print("Using file:", data_path.name)
    return data_path

# =====================
# Step 3 — Load into pandas
# =====================
def step3_load_dataframe(data_path: Path) -> pd.DataFrame:
    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)
    return df

# =====================
# Step 4 — Show first few rows
# =====================
def step4_show_head(df: pd.DataFrame, n: int = 5) -> None:
    print(f"\nFirst {n} rows:")
    print(df.head(n))

# =====================
# Step 5 — Count rows with ANY nulls
# =====================
def step5_count_rows_with_null(df: pd.DataFrame) -> int:
    rows_with_any_null = int((df.isna().sum(axis=1) > 0).sum())
    print(f"\nTotal rows with ≥1 null value: {rows_with_any_null} out of {len(df)}")
    return rows_with_any_null

# =====================
# Step 6 — Count rows where |latitude| <= tol OR |longitude| <= tol
# =====================
def step6_count_zero_lat_lon(df: pd.DataFrame, tol: float = 1e-6) -> None:
    # --- Try common lat/lon column names (case-insensitive) ---
    lat_candidates = ["latitude", "lat", "pickup_latitude", "dropoff_latitude"]
    lon_candidates = ["longitude", "lon", "lng", "pickup_longitude", "dropoff_longitude"]

    def find_col(cands):
        # exact (case-insensitive) first, else substring match as fallback
        cols_lower = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in cols_lower:
                return cols_lower[c]
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in cands):
                return c
        return None

    lat_col = find_col(lat_candidates)
    lon_col = find_col(lon_candidates)

    if not lat_col or not lon_col:
        print("\nLatitude/Longitude columns not found — skipping zero coordinate count.")
        print("Detected columns:", list(df.columns))
        return

    # --- Safely coerce to numeric for comparison ---
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    # --- Use tolerance for near-zero values ---
    lat_near_zero = lat.abs() <= tol
    lon_near_zero = lon.abs() <= tol
    either_near_zero = lat_near_zero | lon_near_zero

    print(f"\nNear-zero coordinate checks with tol={tol} using: lat='{lat_col}', lon='{lon_col}'")
    print(f"Rows with |latitude| <= {tol}: {int(lat_near_zero.sum())}")
    print(f"Rows with |longitude| <= {tol}: {int(lon_near_zero.sum())}")
    print(f"Rows where either |lat|<=tol OR |lon|<=tol: {int(either_near_zero.sum())}")

# =====================
# Step 7 — Data cleaning:
# Drop rows with ANY nulls; then drop rows where latitude==0.0 OR longitude==0.0 (strict)
# =====================
def step7_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    original = len(df)

    # Drop any nulls
    df_no_nulls = df.dropna()
    after_nulls = len(df_no_nulls)

    # Detect lat/lon columns (same logic as Step 6)
    lat_candidates = ["latitude", "lat", "pickup_latitude", "dropoff_latitude"]
    lon_candidates = ["longitude", "lon", "lng", "pickup_longitude", "dropoff_longitude"]

    def find_col(cands):
        cols_lower = {c.lower(): c for c in df_no_nulls.columns}
        for c in cands:
            if c in cols_lower:
                return cols_lower[c]
        for c in df_no_nulls.columns:
            cl = c.lower()
            if any(k in cl for k in cands):
                return c
        return None

    lat_col = find_col(lat_candidates)
    lon_col = find_col(lon_candidates)

    if not lat_col or not lon_col:
        print("\n[Clean] Lat/Lon columns not found — only null rows removed.")
        print(f"Rows: {original} -> {after_nulls} (removed {original - after_nulls})")
        return df_no_nulls

    # Coerce to numeric and filter strict zeros
    lat = pd.to_numeric(df_no_nulls[lat_col], errors="coerce")
    lon = pd.to_numeric(df_no_nulls[lon_col], errors="coerce")

    mask_nonzero = (lat != 0.0) & (lon != 0.0)  # strict equality to zero
    cleaned_df = df_no_nulls[mask_nonzero].copy()

    print(f"\n[Clean] Strict zero filtering on '{lat_col}', '{lon_col}'")
    print(f"Rows: {original} -> {after_nulls} after dropna -> {len(cleaned_df)} after zero-filter")
    print(f"Removed due to nulls: {original - after_nulls}")
    print(f"Removed due to zero lat/lon: {after_nulls - len(cleaned_df)}")
    return cleaned_df

# =====================
# Orchestration
# =====================
def main():
    data_dir = step1_download(DATASET_ID)
    data_path = step2_locate_tabular_file(data_dir)
    df = step3_load_dataframe(data_path)
    step4_show_head(df, n=5)
    step5_count_rows_with_null(df)
    step6_count_zero_lat_lon(df, tol=1e-6)  # diagnostic with tolerance
    cleaned = step7_clean_data(df)          # strict cleaning
    # (Optional) show size of cleaned data
    print(f"\nCleaned DataFrame shape: {cleaned.shape}")

if __name__ == "__main__":
    main()