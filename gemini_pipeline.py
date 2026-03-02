"""
UMI – Agentic Gemini Medical Reasoning Pipeline  (gemini_pipeline.py)
=======================================================================
3-step medical reasoning chain powered by the Gemini Flash API.

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Agent 1 ── AUDITOR                                                 │
  │   Input : JSON of per-hospital weight-divergence scores             │
  │   Output: Outlier verdict + Clinical Hypothesis                     │
  │           (Equipment Variance vs Demographic Shift vs Other)        │
  ├─────────────────────────────────────────────────────────────────────│
  │  Agent 2 ── CLINICIAN                                               │
  │   Input : Top-3 global model coefficients (feature, weight, dir.)   │
  │   Output: 3-sentence research-grade medical brief                   │
  ├─────────────────────────────────────────────────────────────────────│
  │  Agent 3 ── GOVERNANCE                                              │
  │   Input : ε spent, ε budget, δ, mechanism, n_rounds                 │
  │   Output: HIPAA + GDPR compliance verdict + recommendation          │
  └─────────────────────────────────────────────────────────────────────┘

All agents return a structured dict consumable directly by Streamlit components.

run_full_pipeline(...) → {
    "auditor"    : AuditorResult,
    "clinician"  : ClinicianResult,
    "governance" : GovernanceResult,
}
"""

from __future__ import annotations

import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Callable, Optional

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE LABEL GLOSSARY  (for clinical translation)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_GLOSSARY: dict[str, dict] = {
    "ca"      : {"label": "Number of Major Vessels (Fluoroscopy)",
                 "system": "Cardiovascular", "direction": "higher = more disease"},
    "thalch"  : {"label": "Max Heart Rate Achieved (thalch)",
                 "system": "Cardiovascular / Autonomic", "direction": "lower = more risk"},
    "oldpeak" : {"label": "ST-Segment Depression (oldpeak)",
                 "system": "Ischaemic / ECG", "direction": "higher = more ischaemia"},
    "chol"    : {"label": "Serum Cholesterol (mg/dL)",
                 "system": "Metabolic / Lipid", "direction": "higher = more risk"},
    "age"     : {"label": "Patient Age",
                 "system": "Demographic", "direction": "higher = more risk"},
    "trestbps": {"label": "Resting Blood Pressure (mmHg)",
                 "system": "Cardiovascular / Hypertension", "direction": "higher = more risk"},
    "cp"      : {"label": "Chest Pain Type",
                 "system": "Symptomatic / Anginal", "direction": "type 4 = asymptomatic, highest risk"},
    "exang"   : {"label": "Exercise-Induced Angina",
                 "system": "Cardiovascular / Functional", "direction": "1 = present = more risk"},
    "sex"     : {"label": "Patient Sex",
                 "system": "Demographic", "direction": "male = higher baseline risk"},
    "fbs"     : {"label": "Fasting Blood Sugar > 120 mg/dL",
                 "system": "Metabolic / Diabetic", "direction": "1 = elevated = more risk"},
    "restecg" : {"label": "Resting ECG Result",
                 "system": "ECG / Electrical", "direction": "0 = normal"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Regulatory thresholds
# ─────────────────────────────────────────────────────────────────────────────
HIPAA_EPSILON_SAFE      = 1.0    # widely-cited conservative threshold per query
HIPAA_EPSILON_MODERATE  = 3.0    # acceptable with additional safeguards
GDPR_EPSILON_SAFE       = 1.0    # EU Article 5 proportionality equivalent
GDPR_EPSILON_MODERATE   = 2.0    # GDPR allows with DPA notification


# ─────────────────────────────────────────────────────────────────────────────
# Gemini client
# ─────────────────────────────────────────────────────────────────────────────
def _get_client(api_key: str):
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Gemini client init failed: {e}")


def _call_gemini(client, prompt: str,
                 model: str = "gemini-2.0-flash",
                 retries: int = 2) -> str:
    """Call Gemini with auto-retry on transient errors."""
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            if attempt == retries:
                return f"⚠ Gemini API error after {retries + 1} attempts: {e}"
            time.sleep(1.5 * (attempt + 1))


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 1 — AUDITOR
# ═════════════════════════════════════════════════════════════════════════════
def build_divergence_json(
    local_weights  : dict[str, tuple],
    global_weights : np.ndarray,
    feat_cols      : list[str],
) -> dict:
    """
    Compute per-hospital divergence metrics from local vs global weight vectors.

    Returns a JSON-serialisable dict:
    {
        "hospital_name": {
            "l2_divergence": float,
            "cosine_similarity_to_global": float,
            "mean_local_coef": float,
            "std_local_coef": float,
            "top_divergent_feature": str,
            "top_divergent_delta": float,
        },
        ...
    }
    """
    from scipy.spatial.distance import cosine as cosine_dist

    result = {}
    gw = global_weights[:len(feat_cols)]  # align lengths

    for name, (coef, _) in local_weights.items():
        lw = coef[:len(feat_cols)]
        l2  = float(np.linalg.norm(lw - gw))
        cos = float(1.0 - cosine_dist(lw, gw))

        deltas = np.abs(lw - gw)
        top_idx = int(np.argmax(deltas))

        result[name] = {
            "l2_divergence"               : round(l2, 5),
            "cosine_similarity_to_global" : round(cos, 5),
            "mean_local_coef"             : round(float(np.mean(lw)), 5),
            "std_local_coef"              : round(float(np.std(lw)), 5),
            "top_divergent_feature"       : feat_cols[top_idx] if top_idx < len(feat_cols) else "?",
            "top_divergent_delta"         : round(float(deltas[top_idx]), 5),
        }

    return result


def _classify_hypothesis(divergence_json: dict, outlier: str) -> str:
    """
    Heuristic pre-classification of outlier hypothesis to guide Gemini.
    Equipment Variance → outlier's top divergent feature is an ECG/measurement feature
    Demographic Shift  → outlier's top divergent feature is age/sex/geographic
    """
    ECG_MEASUREMENT_FEATURES    = {"oldpeak", "trestbps", "thalch", "restecg"}
    DEMOGRAPHIC_FEATURES        = {"age", "sex", "fbs", "chol"}
    CLINICAL_PROTOCOL_FEATURES  = {"ca", "cp", "exang"}

    feat = divergence_json.get(outlier, {}).get("top_divergent_feature", "")
    if feat in ECG_MEASUREMENT_FEATURES:
        return "Equipment Variance"
    elif feat in DEMOGRAPHIC_FEATURES:
        return "Demographic Shift"
    elif feat in CLINICAL_PROTOCOL_FEATURES:
        return "Clinical Protocol Divergence"
    return "Unknown Cause"


def step1_auditor(
    local_weights  : dict[str, tuple],
    global_weights : np.ndarray,
    feat_cols      : list[str],
    api_key        : str,
    divergence_json: Optional[dict] = None,
) -> dict:
    """
    AUDITOR AGENT — Identifies the statistical outlier hospital and provides
    a clinical hypothesis for why their local model diverges.

    Args:
        local_weights   : {hospital: (coef_1d, intercept_1d)} from federated_core
        global_weights  : 1D global coefficient array
        feat_cols       : Feature column names
        api_key         : Gemini API key
        divergence_json : Pre-computed divergence dict (computed here if None)

    Returns: AuditorResult dict (Streamlit-ready)
    {
        "outlier_hospital"        : str,
        "hypothesis_type"         : "Equipment Variance" | "Demographic Shift" | ...
        "divergence_json"         : dict   ← raw scores per hospital
        "divergence_table"        : pd.DataFrame
        "gemini_verdict"          : str    ← full Gemini response
        "gemini_hypothesis"       : str    ← extracted one-liner hypothesis
        "confidence"              : str    ← "High" / "Medium" / "Low"
        "render"                  : dict   ← pre-formatted strings for Streamlit
    }
    """
    client = _get_client(api_key)

    # ── Compute divergence JSON if not provided ───────────────────────────
    if divergence_json is None:
        divergence_json = build_divergence_json(local_weights, global_weights, feat_cols)

    # ── Find outlier (highest L2 divergence) ─────────────────────────────
    outlier = max(divergence_json, key=lambda h: divergence_json[h]["l2_divergence"])
    hypothesis_type = _classify_hypothesis(divergence_json, outlier)

    # ── Build divergence table ────────────────────────────────────────────
    table_rows = []
    for name, metrics in divergence_json.items():
        table_rows.append({
            "Hospital"             : name,
            "L2 Divergence"        : metrics["l2_divergence"],
            "Cosine Sim (global)"  : metrics["cosine_similarity_to_global"],
            "Top Δ Feature"        : metrics["top_divergent_feature"],
            "Top Δ Value"          : metrics["top_divergent_delta"],
            "Outlier?"             : "🚨 YES" if name == outlier else "✅ No",
        })
    div_table = pd.DataFrame(table_rows).sort_values("L2 Divergence", ascending=False).reset_index(drop=True)

    # ── Format JSON for Gemini ────────────────────────────────────────────
    json_str = json.dumps(divergence_json, indent=2)

    prompt = f"""You are a Senior Federated Learning Auditor for a cross-hospital heart 
disease AI system trained across 4 international medical silos: Cleveland (USA), 
Hungary, Switzerland, and VA Long Beach (USA).

You have received the following JSON of per-hospital weight divergence metrics, 
comparing each hospital's local model against the global FedProx-aggregated model:

```json
{json_str}
```

Preliminary analysis flags **{outlier}** as the statistical outlier with the highest 
L2 divergence. The top divergent feature is '{divergence_json[outlier]['top_divergent_feature']}'.

My pre-classification of the hypothesis type is: **{hypothesis_type}**

In exactly 4 sentences:
1. CONFIRM or REVISE the outlier identification and explain the statistical evidence.
2. State your clinical hypothesis for WHY this hospital diverges — choose and justify 
   the most likely cause from: 'Equipment Variance' (measurement device calibration), 
   'Demographic Shift' (patient age/sex/lifestyle differences), or 
   'Clinical Protocol Divergence' (different diagnostic coding or treatment selection).
3. Quantify the risk this divergence poses to the global model's generalisability.
4. Recommend ONE specific mitigation: either increase μ (proximal penalty), 
   exclude the silo, or request an on-site data audit.

Be technically precise. Use statistical terminology."""

    gemini_verdict = _call_gemini(client, prompt)

    # Extract confidence from cosine similarity
    cos_val = divergence_json[outlier]["cosine_similarity_to_global"]
    confidence = "High" if cos_val < 0.7 else "Medium" if cos_val < 0.9 else "Low"

    # One-liner hypothesis
    gemini_hypothesis = f"{hypothesis_type} — {outlier} shows L2={divergence_json[outlier]['l2_divergence']:.4f} divergence, top feature: {divergence_json[outlier]['top_divergent_feature']}"

    return {
        "outlier_hospital"  : outlier,
        "hypothesis_type"   : hypothesis_type,
        "divergence_json"   : divergence_json,
        "divergence_table"  : div_table,
        "gemini_verdict"    : gemini_verdict,
        "gemini_hypothesis" : gemini_hypothesis,
        "confidence"        : confidence,
        "render"            : {
            "header"       : f"🔍 Auditor — Outlier: **{outlier}**",
            "badge_type"   : hypothesis_type,
            "badge_color"  : {
                "Equipment Variance"         : "#f0883e",
                "Demographic Shift"          : "#a78bfa",
                "Clinical Protocol Divergence": "#f85149",
                "Unknown Cause"              : "#8b949e",
            }.get(hypothesis_type, "#8b949e"),
            "confidence_color": {
                "High": "#f85149", "Medium": "#d29922", "Low": "#3fb950"
            }.get(confidence, "#8b949e"),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 2 — CLINICIAN
# ═════════════════════════════════════════════════════════════════════════════
def step2_clinician(
    global_weights : np.ndarray,
    feat_cols      : list[str],
    api_key        : str,
    top_n          : int = 3,
) -> dict:
    """
    CLINICIAN AGENT — Translates the top-N global model coefficients into a
    research-grade 3-sentence medical brief.

    Returns: ClinicianResult dict (Streamlit-ready)
    {
        "top_features"       : [(feature_name, weight, direction), ...]
        "feature_table"      : pd.DataFrame
        "gemini_brief"       : str   ← 3-sentence research brief
        "gemini_intervention": str   ← actionable clinical recommendation
        "render"             : dict
    }
    """
    client = _get_client(api_key)

    # ── Select top-N features by absolute coefficient ─────────────────────
    gw = global_weights[:len(feat_cols)]
    paired = sorted(zip(feat_cols, gw), key=lambda x: abs(x[1]), reverse=True)
    top_features = [(f, float(w)) for f, w in paired[:top_n]]

    # Build table
    table_rows = []
    for rank, (feat, weight) in enumerate(top_features, 1):
        meta = FEATURE_GLOSSARY.get(feat, {"label": feat, "system": "—", "direction": "—"})
        table_rows.append({
            "Rank"           : rank,
            "Feature"        : feat,
            "Clinical Name"  : meta["label"],
            "Body System"    : meta["system"],
            "Global Weight"  : round(weight, 5),
            "Direction"      : "↑ Risk" if weight > 0 else "↓ Risk",
            "Interpretation" : meta.get("direction", "—"),
        })
    feature_table = pd.DataFrame(table_rows)

    # ── Format for Gemini ─────────────────────────────────────────────────
    feature_lines = "\n".join(
        f"  {rank}. {FEATURE_GLOSSARY.get(f, {}).get('label', f)} (code: '{f}'): "
        f"coefficient = {w:+.5f}  →  {'risk-increasing' if w > 0 else 'risk-reducing'}  "
        f"[{FEATURE_GLOSSARY.get(f, {}).get('system', '?')}]"
        for rank, (f, w) in enumerate(top_features, 1)
    )

    prompt = f"""You are a Senior Cardiologist and Medical AI Researcher. You are writing 
a brief for the New England Journal of Medicine based on findings from a newly developed 
Federated Learning model trained across 4 international hospitals 
(Cleveland USA, Hungary, Switzerland, VA Long Beach USA) using privacy-preserving 
FedProx aggregation with Differential Privacy.

The Unified Global Model's top-3 most predictive features are:

{feature_lines}

Write EXACTLY 3 research-grade sentences:

SENTENCE 1 — Pathophysiology of the #1 feature: Explain its mechanistic link to 
coronary artery disease (CAD) or heart failure, citing the specific body system. 
Reference how the model weight magnitude reflects clinical importance.

SENTENCE 2 — Feature interaction: Explain how features #2 and #3 interact or 
compound cardiovascular risk together, referencing relevant clinical syndromes 
(e.g. Ischaemic Cascade, Metabolic Syndrome, Autonomic Dysfunction).

SENTENCE 3 — Federated learning value: Explain why discovering this feature 
ranking across FOUR geographically and demographically distinct hospital cohorts 
makes this finding more clinically robust than single-centre studies.

Tone: Peer-reviewed journal standard. No bullet points. No headings. Pure prose."""

    gemini_brief = _call_gemini(client, prompt)

    # ── Follow-up: actionable intervention ───────────────────────────────
    intervention_prompt = f"""Based on a heart disease model where the top predictor 
is '{top_features[0][0]}' (weight={top_features[0][1]:+.4f}), give ONE specific, 
evidence-based clinical intervention in a single sentence that a cardiologist 
could act on immediately for a high-risk patient. Be specific with numbers 
(e.g., target values, drug class). No preamble."""

    gemini_intervention = _call_gemini(client, intervention_prompt)

    return {
        "top_features"        : top_features,
        "feature_table"       : feature_table,
        "gemini_brief"        : gemini_brief,
        "gemini_intervention" : gemini_intervention,
        "render"              : {
            "header"         : "🩺 Clinician — Medical Research Brief",
            "top_feature"    : top_features[0][0] if top_features else "—",
            "top_feature_label": FEATURE_GLOSSARY.get(
                top_features[0][0], {}
            ).get("label", top_features[0][0]) if top_features else "—",
            "feature_badges" : [
                {"name": f, "weight": round(w, 4),
                 "color": "#f85149" if w > 0 else "#3fb950"}
                for f, w in top_features
            ],
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 3 — GOVERNANCE
# ═════════════════════════════════════════════════════════════════════════════
def step3_governance(
    epsilon_spent  : float,
    epsilon_budget : float,
    delta          : float,
    n_rounds       : int,
    n_hospitals    : int,
    api_key        : str,
    mechanism      : str = "Gaussian (ε,δ)-DP",
    clip_norm      : float = 1.0,
) -> dict:
    """
    GOVERNANCE AGENT — Evaluates current privacy budget consumption against
    HIPAA Safe Harbor and GDPR Article 5 regulatory thresholds.

    Returns: GovernanceResult dict (Streamlit-ready)
    {
        "hipaa_status"         : "Compliant" | "Marginal" | "Non-Compliant"
        "gdpr_status"          : "Compliant" | "Marginal" | "Non-Compliant"
        "overall_verdict"      : "✅ APPROVED" | "⚠ CONDITIONAL" | "🚨 HALT"
        "budget_remaining"     : float
        "rounds_remaining"     : int    ← estimated safe rounds left
        "gemini_verdict"       : str    ← full Gemini compliance response
        "gemini_recommendation": str
        "render"               : dict
        "compliance_scores"    : dict   ← numeric scores for Streamlit gauges
    }
    """
    client = _get_client(api_key)

    # ── Threshold evaluations ─────────────────────────────────────────────
    budget_remaining  = epsilon_budget - epsilon_spent
    eps_per_round     = epsilon_spent / max(n_rounds, 1)
    rounds_remaining  = int(budget_remaining / eps_per_round) if eps_per_round > 0 else 0

    # HIPAA evaluation
    if epsilon_spent <= HIPAA_EPSILON_SAFE:
        hipaa_status = "Compliant"
        hipaa_detail = f"ε={epsilon_spent:.4f} ≤ {HIPAA_EPSILON_SAFE} (Safe Harbor threshold)"
    elif epsilon_spent <= HIPAA_EPSILON_MODERATE:
        hipaa_status = "Marginal"
        hipaa_detail = f"ε={epsilon_spent:.4f} ∈ ({HIPAA_EPSILON_SAFE}, {HIPAA_EPSILON_MODERATE}] — requires additional safeguards"
    else:
        hipaa_status = "Non-Compliant"
        hipaa_detail = f"ε={epsilon_spent:.4f} > {HIPAA_EPSILON_MODERATE} — exceeds Safe Harbor limit"

    # GDPR evaluation
    if epsilon_spent <= GDPR_EPSILON_SAFE:
        gdpr_status = "Compliant"
        gdpr_detail = f"ε={epsilon_spent:.4f} ≤ {GDPR_EPSILON_SAFE} (Art. 5 proportionality)"
    elif epsilon_spent <= GDPR_EPSILON_MODERATE:
        gdpr_status = "Marginal"
        gdpr_detail = f"ε={epsilon_spent:.4f} ∈ ({GDPR_EPSILON_SAFE}, {GDPR_EPSILON_MODERATE}] — DPA notification recommended"
    else:
        gdpr_status = "Non-Compliant"
        gdpr_detail = f"ε={epsilon_spent:.4f} > {GDPR_EPSILON_MODERATE} — breaches GDPR proportionality principle"

    # Overall verdict
    statuses = {hipaa_status, gdpr_status}
    if "Non-Compliant" in statuses:
        overall_verdict = "🚨 HALT"
        verdict_color   = "#f85149"
    elif "Marginal" in statuses:
        overall_verdict = "⚠ CONDITIONAL"
        verdict_color   = "#d29922"
    else:
        overall_verdict = "✅ APPROVED"
        verdict_color   = "#3fb950"

    # ── Gemini compliance prompt ──────────────────────────────────────────
    prompt = f"""You are a Healthcare Data Privacy Compliance Officer and Legal Advisor 
specialising in AI/ML systems under HIPAA (USA) and GDPR (EU/UK) jurisdictions.

You are reviewing a Differential Privacy deployment in a cross-border Federated 
Learning medical AI system with the following parameters:

SYSTEM PARAMETERS:
  • Privacy Mechanism         : {mechanism}
  • Gradient Clipping Norm    : {clip_norm}
  • Failure Probability (δ)   : {delta}
  • Federated Hospitals       : {n_hospitals}
  • Communication Rounds      : {n_rounds}
  • ε per round (per client)  : {eps_per_round:.4f}
  • Total ε consumed          : {epsilon_spent:.4f}
  • Total ε budget allocated  : {epsilon_budget:.4f}
  • Budget remaining          : {budget_remaining:.4f}
  • Estimated rounds remaining : {rounds_remaining}

PRELIMINARY COMPLIANCE ASSESSMENT:
  • HIPAA (45 CFR §164.514): {hipaa_status} — {hipaa_detail}
  • GDPR (Art. 5 & 9)      : {gdpr_status} — {gdpr_detail}
  • Overall Verdict         : {overall_verdict}

In EXACTLY 3 sentences:
1. HIPAA VERDICT: State compliance status under HIPAA Safe Harbor de-identification 
   standard and cite the specific regulatory risk if marginal or non-compliant.
2. GDPR VERDICT: State compliance status under GDPR Article 5 (data minimisation) 
   and Article 9 (special category health data). Include whether DPA notification 
   is required.
3. RECOMMENDATION: Give one concrete, actionable step to either maintain compliance 
   (if compliant) or urgently remediate the issue (if non-compliant or marginal). 
   Be specific — cite a parameter change, a legal mechanism (e.g., LIA, consent), 
   or an architectural change.

Tone: Legal/regulatory. Precise. No hedging language."""

    gemini_verdict = _call_gemini(client, prompt)

    # One-liner recommendation
    rec_prompt = f"""In ONE sentence (≤25 words), give the single most important 
privacy budget recommendation for a federated learning system where ε_spent={epsilon_spent:.3f} 
and ε_budget={epsilon_budget:.3f}. Be concrete and actionable."""
    gemini_recommendation = _call_gemini(client, rec_prompt)

    # Compliance score as 0–100 scale (for Streamlit gauge/progress)
    compliance_score = max(0.0, min(100.0, (1.0 - epsilon_spent / epsilon_budget) * 100))

    return {
        "hipaa_status"         : hipaa_status,
        "hipaa_detail"         : hipaa_detail,
        "gdpr_status"          : gdpr_status,
        "gdpr_detail"          : gdpr_detail,
        "overall_verdict"      : overall_verdict,
        "epsilon_spent"        : round(epsilon_spent, 5),
        "epsilon_budget"       : epsilon_budget,
        "budget_remaining"     : round(budget_remaining, 5),
        "rounds_remaining"     : rounds_remaining,
        "eps_per_round"        : round(eps_per_round, 5),
        "delta"                : delta,
        "gemini_verdict"       : gemini_verdict,
        "gemini_recommendation": gemini_recommendation,
        "compliance_scores"    : {
            "budget_used_pct"  : round((epsilon_spent / epsilon_budget) * 100, 1),
            "safety_score"     : round(compliance_score, 1),
            "hipaa_score"      : 100 if hipaa_status == "Compliant" else 50 if hipaa_status == "Marginal" else 0,
            "gdpr_score"       : 100 if gdpr_status  == "Compliant" else 50 if gdpr_status  == "Marginal" else 0,
        },
        "render"               : {
            "header"           : "🔒 Governance — Compliance Verdict",
            "verdict"          : overall_verdict,
            "verdict_color"    : verdict_color,
            "hipaa_color"      : "#3fb950" if hipaa_status == "Compliant" else "#d29922" if hipaa_status == "Marginal" else "#f85149",
            "gdpr_color"       : "#3fb950" if gdpr_status  == "Compliant" else "#d29922" if gdpr_status  == "Marginal" else "#f85149",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR — run_full_pipeline()
# ═════════════════════════════════════════════════════════════════════════════
def run_full_pipeline(
    global_weights  : np.ndarray,
    local_weights   : dict[str, tuple],
    feat_cols       : list[str],
    epsilon_spent   : float,
    epsilon_budget  : float,
    n_rounds        : int,
    api_key         : str,
    delta           : float = 1e-5,
    clip_norm       : float = 1.0,
    divergence_json : Optional[dict] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """
    Run all 3 Gemini agents sequentially and return the combined structured output.

    Args:
        global_weights    : 1D array of FedProx-aggregated model coefficients
        local_weights     : {hospital: (coef_1d, intercept_1d)}
        feat_cols         : Feature names in same order as weights
        epsilon_spent     : Total ε consumed across all rounds
        epsilon_budget    : Total ε allocated (e.g. epsilon_per_round × n_rounds)
        n_rounds          : Number of federated rounds completed
        api_key           : Gemini API key
        delta             : DP failure probability (default 1e-5)
        clip_norm         : Gradient clipping norm (default 1.0)
        divergence_json   : Pre-computed divergence dict (auto-computed if None)
        progress_callback : Optional callable(step: int, label: str) for UI updates

    Returns:
        {
            "auditor"     : AuditorResult dict,
            "clinician"   : ClinicianResult dict,
            "governance"  : GovernanceResult dict,
            "meta"        : { "n_hospitals", "n_rounds", "feat_cols", ... }
        }
    """
    results = {}

    # ── Pre-compute divergence JSON (shared across agents) ────────────────
    if divergence_json is None:
        divergence_json = build_divergence_json(local_weights, global_weights, feat_cols)

    # ── Step 1: Auditor ───────────────────────────────────────────────────
    if progress_callback:
        progress_callback(1, "🔍 Auditor — analysing hospital weight divergence…")

    results["auditor"] = step1_auditor(
        local_weights   = local_weights,
        global_weights  = global_weights,
        feat_cols       = feat_cols,
        api_key         = api_key,
        divergence_json = divergence_json,
    )

    # ── Step 2: Clinician ─────────────────────────────────────────────────
    if progress_callback:
        progress_callback(2, "🩺 Clinician — generating research-grade medical brief…")

    results["clinician"] = step2_clinician(
        global_weights = global_weights,
        feat_cols      = feat_cols,
        api_key        = api_key,
        top_n          = 3,
    )

    # ── Step 3: Governance ────────────────────────────────────────────────
    if progress_callback:
        progress_callback(3, "🔒 Governance — verifying HIPAA + GDPR compliance…")

    results["governance"] = step3_governance(
        epsilon_spent  = epsilon_spent,
        epsilon_budget = epsilon_budget,
        delta          = delta,
        n_rounds       = n_rounds,
        n_hospitals    = len(local_weights),
        api_key        = api_key,
        clip_norm      = clip_norm,
    )

    # ── Metadata ──────────────────────────────────────────────────────────
    results["meta"] = {
        "n_hospitals"     : len(local_weights),
        "n_rounds"        : n_rounds,
        "feat_cols"       : feat_cols,
        "epsilon_spent"   : round(epsilon_spent, 5),
        "epsilon_budget"  : epsilon_budget,
        "overall_verdict" : results["governance"]["overall_verdict"],
        "outlier_hospital": results["auditor"]["outlier_hospital"],
        "hypothesis_type" : results["auditor"]["hypothesis_type"],
    }

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST HARNESS
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, textwrap

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("⚠  Set GEMINI_API_KEY env var to test this pipeline.")
        print("   export GEMINI_API_KEY=AIza...")
        raise SystemExit(1)

    # Simulated inputs — Switzerland is a clear outlier
    DUMMY_GLOBAL = np.array([ 0.30, -0.50,  0.20,  0.80, -0.10,  0.60, 0.15, -0.25, 0.05, 0.10, 0.35])
    DUMMY_LOCAL  = {
        "Cleveland"    : (np.array([ 0.31, -0.48,  0.19,  0.79, -0.12,  0.58, 0.14, -0.24, 0.04, 0.09, 0.34]), np.array([0.10])),
        "Hungary"      : (np.array([ 0.28, -0.55,  0.22,  0.83, -0.08,  0.62, 0.17, -0.27, 0.06, 0.12, 0.37]), np.array([0.05])),
        "Switzerland"  : (np.array([ 0.85, -0.90,  0.75,  1.20,  0.40,  0.95, 0.65, -0.80, 0.50, 0.55, 0.88]), np.array([0.30])),
        "VA Long Beach": (np.array([ 0.29, -0.52,  0.21,  0.78, -0.11,  0.60, 0.15, -0.26, 0.05, 0.11, 0.36]), np.array([0.08])),
    }
    FEATS = ["ca", "thalch", "oldpeak", "chol", "age", "trestbps", "cp", "exang", "sex", "fbs", "restecg"]

    print("=" * 70)
    print("  UMI — Gemini 3-Step Medical Reasoning Pipeline  (Test Mode)")
    print("=" * 70)

    def progress(step, label):
        print(f"\n[Step {step}/3] {label}")

    out = run_full_pipeline(
        global_weights  = DUMMY_GLOBAL,
        local_weights   = DUMMY_LOCAL,
        feat_cols       = FEATS,
        epsilon_spent   = 2.5,
        epsilon_budget  = 5.0,
        n_rounds        = 3,
        api_key         = api_key,
        progress_callback = progress,
    )

    # ── Print results ─────────────────────────────────────────────────────
    aud  = out["auditor"]
    clin = out["clinician"]
    gov  = out["governance"]

    print(f"\n{'='*70}")
    print(f"  AGENT 1 — AUDITOR")
    print(f"{'='*70}")
    print(f"  Outlier Hospital  : {aud['outlier_hospital']}")
    print(f"  Hypothesis Type   : {aud['hypothesis_type']}")
    print(f"  Confidence        : {aud['confidence']}")
    print(f"\n  Gemini Verdict:")
    print(textwrap.indent(textwrap.fill(aud['gemini_verdict'], 66), "  "))
    print(f"\n  Divergence Table:")
    print(aud["divergence_table"].to_string(index=False))

    print(f"\n{'='*70}")
    print(f"  AGENT 2 — CLINICIAN")
    print(f"{'='*70}")
    print(f"  Top-3 Features: {[f for f,_ in clin['top_features']]}")
    print(f"\n  Medical Brief:")
    print(textwrap.indent(textwrap.fill(clin['gemini_brief'], 66), "  "))
    print(f"\n  Clinical Intervention:")
    print(textwrap.indent(textwrap.fill(clin['gemini_intervention'], 66), "  "))

    print(f"\n{'='*70}")
    print(f"  AGENT 3 — GOVERNANCE")
    print(f"{'='*70}")
    print(f"  HIPAA Status     : {gov['hipaa_status']}  ({gov['hipaa_detail']})")
    print(f"  GDPR Status      : {gov['gdpr_status']}  ({gov['gdpr_detail']})")
    print(f"  Overall Verdict  : {gov['overall_verdict']}")
    print(f"  ε Remaining      : {gov['budget_remaining']} (≈{gov['rounds_remaining']} more rounds)")
    print(f"\n  Gemini Verdict:")
    print(textwrap.indent(textwrap.fill(gov['gemini_verdict'], 66), "  "))
    print(f"\n  Recommendation   : {gov['gemini_recommendation']}")

    print(f"\n{'='*70}")
    print(f"  META: {out['meta']}")
    print(f"{'='*70}")
