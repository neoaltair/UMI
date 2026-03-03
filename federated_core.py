"""
UMI – Federated Core: FedProx + Differential Privacy + Simulated Homomorphic Encryption
========================================================================================
"""

import base64
import hashlib
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

def _resolve_silo_dir() -> Path:
    for candidate in [Path("data/silos"), Path("silos")]:
        if candidate.exists() and any(candidate.glob("*.csv")):
            return candidate
    return Path("data/silos")

SILO_DIR = _resolve_silo_dir()
SILO_FILES = {
    "Cleveland"    : SILO_DIR / "cleveland.csv",
    "Hungary"      : SILO_DIR / "hungary.csv",
    "Switzerland"  : SILO_DIR / "switzerland.csv",
    "VA Long Beach": SILO_DIR / "long_beach.csv",
}
TARGET_COL   = "num"
RANDOM_STATE = 42
FEATURES     = ["age","trestbps","chol","thalch","oldpeak","ca",
                "sex","cp","fbs","restecg","exang"]
CLIP_NORM    = 1.0


def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df


def load_silo_df(path: Path) -> pd.DataFrame:
    return _encode_df(pd.read_csv(path))


def _compute_sigma(epsilon: float, delta: float,
                   clip_norm: float, n: int) -> float:
    if epsilon <= 0 or delta <= 0 or n == 0:
        return 0.0
    return clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / (epsilon * n)


def _clip_weights(weights: np.ndarray,
                  max_norm: float = CLIP_NORM) -> np.ndarray:
    norm = np.linalg.norm(weights)
    if norm > max_norm:
        weights = weights * (max_norm / norm)
    return weights


def _add_gaussian_noise(weights: np.ndarray, sigma: float,
                        rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return weights
    return weights + rng.normal(0, sigma, size=weights.shape)


# ── Simulated Homomorphic Encryption ─────────────────────────────────────────
class TenSEALCKKSTensor:
    def __init__(self, data: np.ndarray, is_encrypted: bool = True):
        self._data        = data.copy()
        self.is_encrypted = is_encrypted
        self.shape        = data.shape

    def __add__(self, other):
        if isinstance(other, TenSEALCKKSTensor):
            return TenSEALCKKSTensor(self._data + other._data,
                                     is_encrypted=True)
        return NotImplemented

    def __rmul__(self, scalar: float):
        return TenSEALCKKSTensor(self._data * scalar, is_encrypted=True)

    def decrypt(self, secret_key: str = "VALID_KEY") -> np.ndarray:
        if secret_key != "VALID_KEY":
            raise ValueError("Invalid Homomorphic Decryption Key")
        self.is_encrypted = False
        return self._data.copy()

    def __repr__(self):
        if self.is_encrypted:
            h = hashlib.sha256(self._data.tobytes()).hexdigest()
            return (f"<TenSEALCKKSTensor: ENCRYPTED_CIPHERTEXT="
                    f"[{base64.b64encode(h.encode()).decode()[:32]}...]>")
        return f"<TenSEALCKKSTensor: DECRYPTED_PLAINTEXT={self._data}>"


def simulate_he_encrypt(weights: np.ndarray,
                         key=None) -> TenSEALCKKSTensor:
    return TenSEALCKKSTensor(weights, is_encrypted=True)


# ── Local silo training ───────────────────────────────────────────────────────
def train_local_silo(
    X_train, y_train, scaler,
    global_coef=None, global_intercept=None,
    mu=0.01, epsilon=1.0, delta=1e-5,
    n_iter_no_change=5, rng=None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    X_sc      = scaler.transform(X_train)
    n         = len(y_train)
    alpha_reg = mu / (2 * n) if n > 0 else 0.0

    model = SGDClassifier(
        loss="log_loss", alpha=alpha_reg, max_iter=200,
        random_state=RANDOM_STATE, n_iter_no_change=n_iter_no_change,
        early_stopping=False, tol=1e-4,
    )
    model.fit(X_sc, y_train)

    coef      = model.coef_.copy()
    intercept = model.intercept_.copy()

    if global_coef is not None:
        coef      = coef      - mu * (coef      - global_coef)
        intercept = intercept - mu * (intercept - global_intercept)

    coef      = _clip_weights(coef.flatten(), CLIP_NORM).reshape(coef.shape)
    intercept = _clip_weights(intercept, CLIP_NORM)

    sigma       = _compute_sigma(epsilon, delta, CLIP_NORM, n)
    coef_noisy  = _add_gaussian_noise(coef,      sigma, rng)
    intcp_noisy = _add_gaussian_noise(intercept, sigma, rng)

    combined  = np.concatenate([coef_noisy.flatten(), intcp_noisy.flatten()])
    he_cipher = simulate_he_encrypt(combined)

    return {
        "coef"         : coef_noisy,
        "intercept"    : intcp_noisy,
        "he_cipher"    : he_cipher,
        "n_samples"    : n,
        "epsilon_spent": epsilon,
        "sigma"        : sigma,
        "classes"      : model.classes_,
    }


# ── FedProx aggregation ───────────────────────────────────────────────────────
def fedprox_aggregate(silo_results: dict, feat_cols: list,
                       drop_anomalies: bool = True) -> dict:
    silos = list(silo_results.keys())

    if drop_anomalies:
        from anomaly_detection import detect_weight_anomalies
        mock = {n: (r["coef"].flatten(), r["intercept"].flatten())
                for n, r in silo_results.items()}
        report  = detect_weight_anomalies(mock, feat_cols)
        outliers = report["outlier_hospitals"]
        silos   = [n for n in silos if n not in outliers] or silos

    total_n    = sum(silo_results[n]["n_samples"] for n in silos)
    first      = silo_results[silos[0]]
    enc_sum    = (first["n_samples"] / total_n) * first["he_cipher"]

    for name in silos[1:]:
        r       = silo_results[name]
        enc_sum = enc_sum + (r["n_samples"] / total_n) * r["he_cipher"]

    recovered  = enc_sum.decrypt("VALID_KEY")
    n_coef     = int(np.prod(first["coef"].shape))

    return {
        "avg_coef"     : recovered[:n_coef].reshape(first["coef"].shape),
        "avg_intercept": recovered[n_coef:].reshape(first["intercept"].shape),
        "total_samples": total_n,
        "dropped_silos": [n for n in silo_results if n not in silos],
    }


def build_global_model(avg_coef, avg_intercept,
                        classes) -> LogisticRegression:
    gm            = LogisticRegression()
    gm.coef_      = avg_coef
    gm.intercept_ = avg_intercept
    gm.classes_   = classes
    return gm


# ── Multi-round federated training ────────────────────────────────────────────
def run_federated_rounds(
    n_rounds=3, mu=0.01, epsilon=1.0, delta=1e-5,
    test_size=0.10, verbose=True,
) -> dict:

    # Load silos
    silo_data = {}
    for name, path in SILO_FILES.items():
        if not path.exists():
            continue
        df        = load_silo_df(path)
        feat_cols = [c for c in FEATURES if c in df.columns]
        silo_data[name] = (df[feat_cols].copy(), df[TARGET_COL].astype(int),
                           feat_cols)

    if not silo_data:
        raise FileNotFoundError(
            "No silo CSV files found. Run data_preparation.py first.")

    feat_cols = list(silo_data.values())[0][2]

    # Train/test splits
    train_splits = {}
    X_tests, y_tests = [], []
    for name, (X, y, _) in silo_data.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size,
            random_state=RANDOM_STATE, stratify=y)
        train_splits[name] = (X_tr, y_tr)
        X_tests.append(X_te)
        y_tests.append(y_te)

    X_test_all = pd.concat(X_tests, ignore_index=True)
    y_test_all = pd.concat(y_tests, ignore_index=True)

    scaler        = StandardScaler().fit(
        pd.concat([X for X, _ in train_splits.values()], ignore_index=True))
    X_test_scaled = scaler.transform(X_test_all)

    # Federated rounds
    rng              = np.random.default_rng(RANDOM_STATE)
    global_coef      = None
    global_intercept = None
    classes          = None
    round_history    = []
    cumulative_eps   = 0.0
    local_weights_final = {}
    local_accuracies    = {}   # ← NEW: populated on final round
    agg                 = {}

    for rnd in range(1, n_rounds + 1):
        t0           = time.time()
        silo_results = {}

        for name, (X_tr, y_tr) in train_splits.items():
            res = train_local_silo(
                X_tr, y_tr, scaler,
                global_coef=global_coef,
                global_intercept=global_intercept,
                mu=mu, epsilon=epsilon, delta=delta, rng=rng,
            )
            silo_results[name] = res
            if classes is None:
                classes = res["classes"]

            # ── Compute local accuracy on final round only ────────────────
            if rnd == n_rounds:
                local_m = build_global_model(
                    res["coef"], res["intercept"], res["classes"])
                local_accuracies[name] = round(
                    accuracy_score(y_test_all,
                                   local_m.predict(X_test_scaled)) * 100, 2)

        agg              = fedprox_aggregate(silo_results, feat_cols,
                                              drop_anomalies=True)
        global_coef      = agg["avg_coef"]
        global_intercept = agg["avg_intercept"]

        gm      = build_global_model(global_coef, global_intercept, classes)
        acc     = accuracy_score(y_test_all, gm.predict(X_test_scaled))
        cumulative_eps += epsilon
        elapsed = time.time() - t0

        round_history.append({
            "Round"        : rnd,
            "Accuracy"     : round(acc * 100, 2),
            "Epsilon_Round": round(epsilon, 4),
            "Epsilon_Spent": round(cumulative_eps, 4),
            "Time_s"       : round(elapsed, 2),
        })

        if verbose:
            print(f"  Round {rnd}/{n_rounds} | "
                  f"Accuracy: {acc*100:.2f}% | "
                  f"ε_spent: {cumulative_eps:.4f} | {elapsed:.2f}s")

        local_weights_final = {
            n: (r["coef"].flatten(), r["intercept"].flatten())
            for n, r in silo_results.items()
        }

    return {
        "global_model"    : build_global_model(global_coef, global_intercept,
                                               classes),
        "scaler"          : scaler,
        "feat_cols"       : feat_cols,
        "local_weights"   : local_weights_final,
        "round_history"   : pd.DataFrame(round_history),
        "X_test_all"      : X_test_all,
        "y_test_all"      : y_test_all,
        "silo_sizes"      : {n: len(y) for n,(_, y) in train_splits.items()},
        "local_accuracies": local_accuracies,          # ← NEW
        "dropped_silos"   : agg.get("dropped_silos", []),  # ← NEW
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  UMI — FedProx + DP + Simulated HE Engine")
    print("=" * 60)
    result = run_federated_rounds(n_rounds=3, mu=0.01,
                                   epsilon=1.0, verbose=True)
    hist   = result["round_history"]
    print("\n  Round History:")
    print(hist.to_string(index=False))
    print(f"\n  Final Accuracy  : {hist['Accuracy'].iloc[-1]:.2f}%")
    print(f"  ε consumed      : {hist['Epsilon_Spent'].iloc[-1]:.4f}")
    print(f"  Local Accuracies: {result['local_accuracies']}")
    print("=" * 60)
