"""
UMI – Federated Learning Data Preparation
==========================================
Steps:
  1. Authenticate with Kaggle API & download 'redwankarimsony/heart-disease-data'
  2. Fill missing values with the column median
  3. Binarise 'num': 0 = no disease, 1 = any stage (1–4)
  4. Split by 'dataset' into 4 hospital CSVs (drop 'id' & 'dataset')
  5. Print row counts per hospital to confirm the split
"""

import os
import json
import zipfile
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
KAGGLE_JSON  = Path.home() / ".kaggle" / "kaggle.json"   # standard location
KAGGLE_SLUG  = "redwankarimsony/heart-disease-data"
DOWNLOAD_DIR = Path("data/raw")
SILO_DIR     = Path("data/silos")

# Exact mapping: unique values in 'dataset' column → output filename
DATASET_MAP = {
    "Cleveland"   : "cleveland.csv",
    "Hungary"     : "hungary.csv",
    "Switzerland" : "switzerland.csv",
    "VA Long Beach": "long_beach.csv",
}

# ── 1.  Kaggle Download ─────────────────────────────────────────────────────
def download_dataset():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Set env vars so the kaggle package picks up credentials automatically
    if KAGGLE_JSON.exists():
        creds = json.loads(KAGGLE_JSON.read_text())
        os.environ.setdefault("KAGGLE_USERNAME", creds["username"])
        os.environ.setdefault("KAGGLE_KEY",      creds["key"])
    else:
        raise FileNotFoundError(
            f"kaggle.json not found at {KAGGLE_JSON}.\n"
            "Download it from https://www.kaggle.com/settings and place it there."
        )

    try:
        import kaggle
    except ImportError:
        raise ImportError("Run:  pip install kaggle")

    print(f"[1/4]  Downloading '{KAGGLE_SLUG}' …")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset=KAGGLE_SLUG,
        path=str(DOWNLOAD_DIR),
        unzip=True,
        quiet=False,
    )
    print("       Done.\n")


def find_csv() -> Path:
    """Return the first (preferably 'heart'-named) CSV in the download dir."""
    csvs = list(DOWNLOAD_DIR.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found under {DOWNLOAD_DIR}")
    preferred = [f for f in csvs if "heart" in f.name.lower()]
    return preferred[0] if preferred else csvs[0]


# ── 2.  Load & impute ───────────────────────────────────────────────────────
def load_and_impute(csv_path: Path) -> pd.DataFrame:
    print(f"[2/4]  Loading  →  {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"       Shape: {df.shape}")

    # Fill ALL missing values with the column median (numeric columns only)
    before = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    after = df.isnull().sum().sum()
    print(f"       Missing values: {before} → {after} (filled with median)\n")
    return df


# ── 3.  Binarise target ─────────────────────────────────────────────────────
def binarise_target(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/4]  Binarising 'num' column …")
    df["num"] = (df["num"] > 0).astype(int)
    counts = df["num"].value_counts().sort_index().to_dict()
    print(f"       0 (no disease): {counts.get(0, 0)}  |  "
          f"1 (disease present): {counts.get(1, 0)}\n")
    return df


# ── 4 & 5.  Split into silos ────────────────────────────────────────────────
def split_into_silos(df: pd.DataFrame):
    SILO_DIR.mkdir(parents=True, exist_ok=True)

    print("[4/4]  Splitting into hospital silos …")
    print(f"       Unique 'dataset' values: {sorted(df['dataset'].unique())}\n")

    results = {}
    for dataset_val, filename in DATASET_MAP.items():
        subset = df[df["dataset"] == dataset_val].copy()

        if subset.empty:
            print(f"  ⚠  No rows found for '{dataset_val}' – skipping.")
            continue

        # Drop columns not needed for training
        subset.drop(columns=["id", "dataset"], inplace=True, errors="ignore")

        out_path = SILO_DIR / filename
        subset.to_csv(out_path, index=False)
        results[dataset_val] = len(subset)
        print(f"  ✔  {dataset_val:<18} → {filename:<20}  ({len(subset):>3} rows)")

    # Confirm total
    total = sum(results.values())
    print(f"\n       Total rows across silos: {total}")
    return results


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    download_dataset()
    csv_path = find_csv()
    df = load_and_impute(csv_path)
    df = binarise_target(df)
    row_counts = split_into_silos(df)

    print("\n" + "=" * 50)
    print("  Data preparation complete!")
    print("  Hospital silo sizes:")
    for hospital, rows in row_counts.items():
        print(f"    {hospital:<20} : {rows} rows")
    print("=" * 50)


if __name__ == "__main__":
    main()
