"""
UMI — Unified Medical Intelligence  |  Streamlit Command Center
================================================================
RBAC:  🩺 Doctor  |  🔬 Researcher  |  🔒 Regulator
Fixed: no extra ML in app.py, single anomaly call, lazy panels,
       no duplicate xaxis/yaxis in update_layout calls.
"""

import datetime, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="UMI – Unified Medical Intelligence",
                   page_icon="🫀", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#070b12 0%,#0d1117 50%,#111820 100%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1117,#070b12);border-right:1px solid #21262d;}
.kpi{background:linear-gradient(135deg,#161b22,#1c2230);border:1px solid #21262d;border-radius:14px;padding:18px 20px;text-align:center;transition:transform .2s,box-shadow .25s;}
.kpi:hover{transform:translateY(-3px);box-shadow:0 12px 30px rgba(0,0,0,.5);}
.kpi-label{color:#6e7681;font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px;}
.kpi-value{color:#e6edf3;font-size:1.8rem;font-weight:800;}
.kpi-sub{color:#3fb950;font-size:.7rem;margin-top:4px;font-weight:600;}
.kpi-sub-blue{color:#58a6ff;font-size:.7rem;margin-top:4px;}
.sec{font-size:1.2rem;font-weight:700;color:#e6edf3;border-bottom:2px solid #1f6feb;padding-bottom:8px;margin-bottom:16px;}
.gbox{background:linear-gradient(135deg,#0d1117,#0f1923);border:1px solid #1f6feb;border-left:4px solid #1f6feb;border-radius:12px;padding:20px 24px;margin-top:20px;}
.gtitle{color:#58a6ff;font-weight:700;font-size:.8rem;letter-spacing:.07em;text-transform:uppercase;margin-bottom:10px;}
.gtext{color:#c9d1d9;font-size:.9rem;line-height:1.75;}
.cipher{background:#0d1117;border:1px solid #21262d;border-left:3px solid #7c3aed;border-radius:8px;padding:10px 14px;font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#6e7681;word-break:break-all;max-height:60px;overflow:hidden;}
.he-badge{display:inline-block;background:#1f3d5c22;border:1px solid #2563eb44;border-radius:20px;padding:2px 10px;font-size:.68rem;color:#58a6ff;font-weight:700;}
.dp-badge{display:inline-block;background:#3d1f5c22;border:1px solid #7c3aed44;border-radius:20px;padding:2px 10px;font-size:.68rem;color:#a78bfa;font-weight:700;}
.feed-container{background:#0d1117;border:1px solid #21262d;border-radius:12px;padding:16px;max-height:400px;overflow-y:auto;font-family:'JetBrains Mono',monospace;}
.feed-line{font-size:.72rem;line-height:1.9;border-bottom:1px solid #161b22;padding:3px 0;}
.feed-ts{color:#3d444d;}
.feed-ok{color:#3fb950;}
.feed-warn{color:#d29922;}
.feed-info{color:#58a6ff;}
.feed-he{color:#a78bfa;}
.feed-dp{color:#79c0ff;}
.arch-card{background:linear-gradient(135deg,#161b22,#1a2030);border:1px solid #30363d;border-radius:12px;padding:16px;text-align:center;transition:transform .2s;}
.arch-card:hover{transform:translateY(-2px);border-color:#58a6ff;}
.arch-hospital{color:#6e7681;font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;}
.arch-model{color:#e6edf3;font-size:.9rem;font-weight:700;margin-bottom:8px;}
.arch-acc{font-size:1.4rem;font-weight:800;}
.arch-meta{color:#6e7681;font-size:.68rem;margin-top:4px;}
.arch-badge{display:inline-block;border-radius:6px;padding:2px 8px;font-size:.65rem;font-weight:700;margin-top:6px;}
.wh-box{background:linear-gradient(135deg,#0f1923,#111820);border:1px solid #238636;border-left:4px solid #3fb950;border-radius:12px;padding:18px 22px;margin:12px 0;}
.wh-title{color:#3fb950;font-weight:700;font-size:.78rem;letter-spacing:.07em;text-transform:uppercase;margin-bottom:10px;}
.wh-item{color:#c9d1d9;font-size:.88rem;line-height:1.8;display:flex;gap:10px;margin-bottom:4px;}
.def-row{display:flex;align-items:center;gap:12px;padding:10px 14px;border-radius:8px;margin-bottom:8px;border:1px solid #21262d;}
.def-pass{background:#0d2b1a;border-color:#238636;}
.def-fail{background:#2b0d0d;border-color:#da3633;}
.def-hospital{color:#e6edf3;font-weight:700;font-size:.85rem;flex:1;}
.def-metrics{color:#6e7681;font-size:.72rem;flex:2;}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
def _resolve_silo_dir():
    for c in [Path("data/silos"), Path("silos")]:
        if c.exists() and any(c.glob("*.csv")):
            return c
    return Path("data/silos")

SILO_DIR   = _resolve_silo_dir()
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
SILO_COLORS  = {"Cleveland":"#58a6ff","Hungary":"#3fb950",
                "Switzerland":"#f0883e","VA Long Beach":"#a78bfa"}
SILO_FLAGS   = {"Cleveland":"🇺🇸","Hungary":"🇭🇺",
                "Switzerland":"🇨🇭","VA Long Beach":"🇺🇸"}

def data_available():
    return any(p.exists() for p in SILO_FILES.values())

# ── Cached helpers ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_silo(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

@st.cache_data(show_spinner=False)
def load_all() -> pd.DataFrame:
    frames = []
    for name, path in SILO_FILES.items():
        if path.exists():
            df = load_silo(str(path))
            df["_hospital"] = name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_resource(show_spinner=False)
def run_fl(n_rounds, mu, epsilon):
    from federated_core import run_federated_rounds
    return run_federated_rounds(n_rounds=n_rounds, mu=mu, epsilon=epsilon,
                                delta=1e-5, test_size=0.10, verbose=False)

# ── Federation log builder (pure formatting, zero ML) ────────────────────────
def build_feed_html(fl, n_rounds, mu, epsilon, anom_result, silo_patients):
    lw       = fl["local_weights"]
    rh       = fl["round_history"]
    outliers = anom_result["outlier_hospitals"]
    anom_df  = anom_result["anomaly_report"]

    KIND = {"ok":("feed-ok","✓"), "info":("feed-info","◈"),
            "warn":("feed-warn","⚠"), "he":("feed-he","🔐"),
            "dp":("feed-dp","🛡")}

    def L(kind, msg):
        cls, icon = KIND.get(kind, ("feed-info","·"))
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        return (f"<div class='feed-line'><span class='feed-ts'>[{ts}]</span> "
                f"<span class='{cls}'>{icon} {msg}</span></div>")

    html  = L("info","═══ UMI Federated Learning Session ═══")
    html += L("info",f"Config: {n_rounds} rounds · μ={mu} · ε={epsilon} · δ=1e-5")
    html += L("info",f"Silos: {len(lw)}/4 online · Dataset: UCI Heart Disease")

    for rnd in range(1, n_rounds+1):
        row  = rh[rh["Round"]==rnd]
        acc  = float(row["Accuracy"].values[0]) if len(row) else 0.0
        eps  = float(row["Epsilon_Spent"].values[0]) if len(row) else 0.0
        html += L("info", f"─── Round {rnd}/{n_rounds} ───")

        for name in lw:
            flag   = SILO_FLAGS.get(name,"🏥")
            coef   = lw[name][0]
            n_pts  = silo_patients.get(name, 0)
            l2     = float(np.linalg.norm(coef))
            sigma  = (CLIP_NORM * np.sqrt(2*np.log(1.25/1e-5))
                      / (epsilon * max(n_pts,1)))
            html += L("ok",  f"{flag} {name} [{n_pts} pts] → FedProx train (μ={mu})")
            if mu > 0:
                html += L("dp", f"  ↳ Proximal: w ← w - {mu}·(w - w_global)")
            html += L("dp",  f"  ↳ Clip: L2={l2:.4f} → ≤1.0")
            html += L("dp",  f"  ↳ Gaussian noise: σ={sigma:.6f} (ε={epsilon}, n={n_pts})")
            html += L("he",  f"  ↳ CKKS encrypt → ciphertext [poly=8192, scale=2^40]")
            is_out = name in outliers
            if is_out:
                r_a = anom_df[anom_df["Hospital"]==name]
                d2  = float(r_a["Mahalanobis D²"].values[0]) if len(r_a) else 0.0
                html += L("warn", f"  ↳ ⚠ D²={d2:.4f} → OUTLIER FLAGGED → DROPPED")
            else:
                html += L("ok", f"  ↳ ✓ Anomaly
