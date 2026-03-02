"""
UMI – Federated Learning (FedAvg) Pipeline
===========================================
Steps:
  1. Reserve a 10% test set from each silo (combined test set)
  2. Train a local LogisticRegression on each hospital's training split
  3. Aggregate model weights via Federated Averaging (FedAvg)
  4. Re-apply averaged weights to a Global Model
  5. Compare Global Model vs. Cleveland-only Local Model on the combined test set
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────────
# Support both data/silos/ (created by data_preparation.py) and a top-level silos/ folder
def _resolve_silo_dir() -> Path:
    for candidate in [Path("data/silos"), Path("silos")]:
        if candidate.exists() and any(candidate.glob("*.csv")):
            return candidate
    return Path("data/silos")   # default; will show a clear error if missing

SILO_DIR = _resolve_silo_dir()

SILO_FILES = {
    "Cleveland"   : SILO_DIR / "cleveland.csv",
    "Hungary"      : SILO_DIR / "hungary.csv",
    "Switzerland" : SILO_DIR / "switzerland.csv",
    "VA Long Beach": SILO_DIR / "long_beach.csv",
}

TARGET_COL = "num"
TEST_SIZE  = 0.10   # 10% held out per silo for the combined test set
RANDOM_STATE = 42

# ── Helper: load one silo and split ─────────────────────────────────────────
def load_silo(file_path: Path):
    """Load CSV, return (X, y) with all numeric columns."""
    df = pd.read_csv(file_path)

    # Fill any residual NaNs with column median
    df.fillna(df.median(numeric_only=True), inplace=True)

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL], errors="ignore")

    # Keep only numeric features
    X = X.select_dtypes(include="number")
    return X, y


# ── Step 1: build combined test set ─────────────────────────────────────────
def build_train_test_splits():
    """
    Reserve TEST_SIZE fraction from each silo.
    Returns:
      train_splits – dict {name: (X_train, y_train)}
      X_test_all   – combined test features (numpy)
      y_test_all   – combined test labels (numpy)
    """
    train_splits = {}
    X_tests, y_tests = [], []

    for name, path in SILO_FILES.items():
        X, y = load_silo(path)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        train_splits[name] = (X_tr, y_tr)
        X_tests.append(X_te)
        y_tests.append(y_te)

    X_test_all = pd.concat(X_tests, ignore_index=True)
    y_test_all = pd.concat(y_tests, ignore_index=True)

    print(f"Combined test set: {len(y_test_all)} samples "
          f"({y_test_all.mean()*100:.1f}% positive)\n")
    return train_splits, X_test_all, y_test_all


# ── Step 2: train one local model, return weights ───────────────────────────
def get_local_model_weights(X_train: pd.DataFrame, y_train: pd.Series,
                             scaler: StandardScaler):
    """
    Train a LogisticRegression on one hospital's data.
    Returns (coef_, intercept_) – the raw weight vectors.
    """
    X_scaled = scaler.transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_scaled, y_train)
    return model.coef_.copy(), model.intercept_.copy()


# ── Step 3: Federated Averaging ──────────────────────────────────────────────
def federated_average(local_weights: list[tuple]) -> tuple:
    """
    FedAvg: simple (unweighted) average of all clients' coef_ and intercept_.
    local_weights: list of (coef_, intercept_) tuples
    Returns: (avg_coef, avg_intercept)
    """
    all_coefs       = np.array([w[0] for w in local_weights])
    all_intercepts  = np.array([w[1] for w in local_weights])
    return all_coefs.mean(axis=0), all_intercepts.mean(axis=0)


# ── Step 4: apply averaged weights to a fresh model object ──────────────────
def build_global_model(avg_coef: np.ndarray, avg_intercept: np.ndarray,
                        reference_model: LogisticRegression) -> LogisticRegression:
    """Clone a trained model and overwrite its weights with the FedAvg result."""
    global_model             = LogisticRegression()
    global_model.coef_       = avg_coef
    global_model.intercept_  = avg_intercept
    global_model.classes_    = reference_model.classes_
    return global_model


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  UMI – Federated Averaging (FedAvg) Pipeline")
    print("=" * 60 + "\n")

    # ── 1. Splits ────────────────────────────────────────────────────────────
    train_splits, X_test_all, y_test_all = build_train_test_splits()

    # ── 2. Fit a shared scaler on ALL training data combined ─────────────────
    #       (in real FL this would be approximated; for POC we use a global one)
    X_all_train = pd.concat([X for X, _ in train_splits.values()], ignore_index=True)
    scaler = StandardScaler().fit(X_all_train)
    X_test_scaled = scaler.transform(X_test_all)

    # ── 3. Local training – collect weights from each hospital ───────────────
    print("Local Training Round")
    print("-" * 40)
    local_weights  = []
    reference_model = None

    for name, (X_tr, y_tr) in train_splits.items():
        coef, intercept = get_local_model_weights(X_tr, y_tr, scaler)
        local_weights.append((coef, intercept))

        # Quick per-hospital local accuracy (for info only)
        tmp = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        tmp.fit(scaler.transform(X_tr), y_tr)
        local_acc = accuracy_score(y_test_all, tmp.predict(X_test_scaled))
        print(f"  {name:<18} | weights shape: {coef.shape} "
              f"| accuracy on combined test: {local_acc*100:.2f}%")

        if name == "Cleveland":
            cleveland_model = tmp           # save for comparison

        if reference_model is None:
            reference_model = tmp           # any model to borrow .classes_

    # ── 4. FedAvg Aggregation ────────────────────────────────────────────────
    print("\nFederated Averaging …")
    avg_coef, avg_intercept = federated_average(local_weights)
    global_model = build_global_model(avg_coef, avg_intercept, reference_model)
    print("  Global model weights computed.\n")

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    global_preds    = global_model.predict(X_test_scaled)
    cleveland_preds = cleveland_model.predict(X_test_scaled)

    global_acc    = accuracy_score(y_test_all, global_preds)
    cleveland_acc = accuracy_score(y_test_all, cleveland_preds)

    print("=" * 60)
    print("  RESULTS – Unified Intelligence vs. Local Silo")
    print("=" * 60)
    print(f"  Unified Global Accuracy  : {global_acc*100:.2f}%")
    print(f"  Local Cleveland Accuracy : {cleveland_acc*100:.2f}%")
    delta = (global_acc - cleveland_acc) * 100
    sign  = "+" if delta >= 0 else ""
    print(f"  Federated Advantage      : {sign}{delta:.2f}%")
    print("=" * 60)

    print("\n  Hospital silo test-set breakdown:")
    print(f"  {'Hospital':<18} | Rows")
    print(f"  {'-'*28}")
    for name, path in SILO_FILES.items():
        X, y = load_silo(path)
        n_test = int(len(y) * TEST_SIZE)
        print(f"  {name:<18} | ~{n_test}")

    print("\nDone.")


if __name__ == "__main__":
    main()
