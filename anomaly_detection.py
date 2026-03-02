"""
UMI – Federated Anomaly Detection
===================================
Identifies statistical outliers at the hospital level by analysing
local model weight vectors. Two complementary methods:

  1. Cosine Similarity Matrix  : pairwise alignment between hospital weight vectors
  2. Mahalanobis Distance      : deviation of each hospital from the centroid
     (flags hospitals with 'noisy equipment' or unusual patient distributions)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import chi2

# ── Main: detect weight anomalies ─────────────────────────────────────────────
def detect_weight_anomalies(
    local_weights: dict[str, tuple],
    feat_cols: list[str],
    alpha: float = 0.05,          # significance level for χ² Mahalanobis test
    zscore_threshold: float = 2.0,
) -> dict:
    """
    Analyse local model weight vectors to flag statistical outlier hospitals.

    Args:
        local_weights      : {hospital: (coef_1d, intercept_1d)}
        feat_cols          : Feature column names (same order as coef)
        alpha              : Significance level for Mahalanobis χ² test  
        zscore_threshold   : L2 Z-score threshold for outlier flag

    Returns:
        dict with:
            anomaly_report   : pd.DataFrame (per-hospital metrics + anomaly flag)
            cosine_sim_matrix: pd.DataFrame (pairwise cosine similarities)
            outlier_hospitals: list of flagged hospital names
            summary          : human-readable string
    """
    hospitals = list(local_weights.keys())
    n         = len(hospitals)

    if n < 2:
        return {
            "anomaly_report"   : pd.DataFrame(),
            "cosine_sim_matrix": pd.DataFrame(),
            "outlier_hospitals": [],
            "summary"          : "Need ≥ 2 hospitals for anomaly detection.",
        }

    # Extract feature-length coefficient vectors
    coef_matrix = []
    for name in hospitals:
        coef = local_weights[name][0]
        flat = coef[:len(feat_cols)]    # take only feature coefficients
        coef_matrix.append(flat)

    coef_matrix = np.array(coef_matrix)   # shape: (n_hospitals, n_features)
    centroid    = coef_matrix.mean(axis=0)

    # ── 1. Cosine Similarity Matrix ─────────────────────────────────────────
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = 1.0 - cosine(coef_matrix[i], coef_matrix[j])

    cosine_df = pd.DataFrame(sim_matrix, index=hospitals, columns=hospitals)

    # ── 2. Mahalanobis Distance ────────────────────────────────────────────
    # Covariance with regularisation (avoids singular matrix)
    cov = np.cov(coef_matrix.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.eye(cov_reg.shape[0])

    mahal_distances = []
    for vec in coef_matrix:
        diff = vec - centroid
        d2   = float(diff @ cov_inv @ diff)
        mahal_distances.append(max(d2, 0.0))

    # χ² test: d² ~ χ²(p) under null hypothesis
    p_values = [1 - chi2.cdf(d2, df=coef_matrix.shape[1])
                for d2 in mahal_distances]

    # ── 3. L2 Z-score anomaly flag ─────────────────────────────────────────
    l2_norms   = [np.linalg.norm(vec - centroid) for vec in coef_matrix]
    l2_mean    = np.mean(l2_norms)
    l2_std     = np.std(l2_norms) + 1e-9
    l2_zscores = [(d - l2_mean) / l2_std for d in l2_norms]

    # Mean cosine similarity to others (lower = more different)
    mean_cosine = []
    for i in range(n):
        others = [sim_matrix[i, j] for j in range(n) if j != i]
        mean_cosine.append(np.mean(others) if others else 1.0)

    # Build report
    report_rows = []
    outlier_hospitals = []
    for idx, name in enumerate(hospitals):
        is_mahal_outlier = p_values[idx] < alpha
        is_z_outlier     = l2_zscores[idx] > zscore_threshold
        is_outlier       = is_mahal_outlier or is_z_outlier

        if is_outlier:
            outlier_hospitals.append(name)

        interpretation = (
            "🚨 Statistical Outlier — data distribution significantly diverges from peers"
            if is_outlier else
            "✅ Normal — weight vector consistent with global distribution"
        )
        report_rows.append({
            "Hospital"        : name,
            "L2 Distance"     : round(l2_norms[idx], 4),
            "L2 Z-Score"      : round(l2_zscores[idx], 3),
            "Mahalanobis D²"  : round(mahal_distances[idx], 4),
            "χ² p-value"      : round(p_values[idx], 4),
            "Mean Cosine Sim" : round(mean_cosine[idx], 4),
            "Is Anomaly"      : "🚨 Yes" if is_outlier else "✅ No",
            "Interpretation"  : interpretation,
        })

    anomaly_df = pd.DataFrame(report_rows).sort_values(
        "L2 Z-Score", ascending=False
    ).reset_index(drop=True)

    # Summary string
    if outlier_hospitals:
        summary = (
            f"⚠ Anomaly detected: **{', '.join(outlier_hospitals)}** "
            f"show statistically significant divergence from the federated consensus. "
            f"This may indicate non-IID data distributions, measurement noise, or "
            f"unique patient cohort characteristics."
        )
    else:
        summary = (
            "✅ All hospital weight vectors are statistically consistent. "
            "No anomalous data distributions detected across the federated network."
        )

    return {
        "anomaly_report"   : anomaly_df,
        "cosine_sim_matrix": cosine_df,
        "outlier_hospitals": outlier_hospitals,
        "summary"          : summary,
    }


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate weights: Switzerland is an obvious outlier
    local_w = {
        "Cleveland"    : (np.array([0.31, -0.48, 0.19, 0.79, -0.12, 0.58]), np.array([0.1])),
        "Hungary"      : (np.array([0.28, -0.55, 0.22, 0.83, -0.08, 0.62]), np.array([0.05])),
        "Switzerland"  : (np.array([1.85, -1.90, 1.70, 2.20,  1.40, 1.95]), np.array([0.9])),
        "VA Long Beach": (np.array([0.29, -0.52, 0.21, 0.78, -0.11, 0.60]), np.array([0.08])),
    }
    feats = ["age", "thalch", "oldpeak", "ca", "chol", "trestbps"]
    result = detect_weight_anomalies(local_w, feats)
    print("\nAnomaly Report:")
    print(result["anomaly_report"].to_string(index=False))
    print(f"\nSummary: {result['summary']}")
    print("\nCosine Similarity Matrix:")
    print(result["cosine_sim_matrix"].round(3))
