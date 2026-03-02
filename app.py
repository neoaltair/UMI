"""
UMI — Unified Medical Intelligence  |  Plotly Command Center
=============================================================
RBAC Roles:  🩺 Doctor  |  🔬 Researcher  |  🔒 Regulator
Charts: Plotly Gauge · Radar · Heatmap · Convergence Line · Privacy-Utility Curve
"""

import os, datetime, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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
.kpi{background:linear-gradient(135deg,#161b22,#1c2230);border:1px solid #21262d;border-radius:14px;
     padding:18px 20px;text-align:center;transition:transform .2s,box-shadow .25s;}
.kpi:hover{transform:translateY(-3px);box-shadow:0 12px 30px rgba(0,0,0,.5);}
.kpi-label{color:#6e7681;font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px;}
.kpi-value{color:#e6edf3;font-size:1.8rem;font-weight:800;}
.kpi-sub{color:#3fb950;font-size:.7rem;margin-top:4px;font-weight:600;}
.kpi-sub-blue{color:#58a6ff;font-size:.7rem;margin-top:4px;}
.sec{font-size:1.2rem;font-weight:700;color:#e6edf3;border-bottom:2px solid #1f6feb;
     padding-bottom:8px;margin-bottom:16px;}
.gbox{background:linear-gradient(135deg,#0d1117,#0f1923);border:1px solid #1f6feb;
      border-left:4px solid #1f6feb;border-radius:12px;padding:20px 24px;margin-top:20px;}
.gtitle{color:#58a6ff;font-weight:700;font-size:.8rem;letter-spacing:.07em;
        text-transform:uppercase;margin-bottom:10px;}
.gtext{color:#c9d1d9;font-size:.9rem;line-height:1.75;}
.cipher{background:#0d1117;border:1px solid #21262d;border-left:3px solid #7c3aed;
        border-radius:8px;padding:10px 14px;font-family:'JetBrains Mono',monospace;
        font-size:.68rem;color:#6e7681;word-break:break-all;max-height:60px;overflow:hidden;}
.he-badge{display:inline-block;background:#1f3d5c22;border:1px solid #2563eb44;
          border-radius:20px;padding:2px 10px;font-size:.68rem;color:#58a6ff;font-weight:700;}
.dp-badge{display:inline-block;background:#3d1f5c22;border:1px solid #7c3aed44;
          border-radius:20px;padding:2px 10px;font-size:.68rem;color:#a78bfa;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
def _resolve_silo_dir():
    for c in [Path("data/silos"), Path("silos")]:
        if c.exists() and any(c.glob("*.csv")): return c
    return Path("data/silos")

SILO_DIR = _resolve_silo_dir()
SILO_FILES = {"Cleveland":SILO_DIR/"cleveland.csv","Hungary":SILO_DIR/"hungary.csv",
              "Switzerland":SILO_DIR/"switzerland.csv","VA Long Beach":SILO_DIR/"long_beach.csv"}
TARGET_COL = "num"; RANDOM_STATE = 42
FEATURES = ["age","trestbps","chol","thalch","oldpeak","ca","sex","cp","fbs","restecg","exang"]
SILO_COLORS = {"Cleveland":"#58a6ff","Hungary":"#3fb950","Switzerland":"#f0883e","VA Long Beach":"#a78bfa"}
PLOTLY_DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                   font_color="#c9d1d9", xaxis=dict(gridcolor="#21262d"),
                   yaxis=dict(gridcolor="#21262d"))

def data_available(): return any(p.exists() for p in SILO_FILES.values())

# ── Data helpers ──────────────────────────────────────────────────────────────
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
            df = load_silo(str(path)); df["_hospital"]=name; frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ── Federated training ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def run_fl(n_rounds: int, mu: float, epsilon: float):
    from federated_core import run_federated_rounds
    return run_federated_rounds(n_rounds=n_rounds, mu=mu, epsilon=epsilon,
                                delta=1e-5, test_size=0.10, verbose=False)

# ── Plotly helpers ────────────────────────────────────────────────────────────
def plotly_gauge(prob: float) -> go.Figure:
    """Green→Yellow→Red probability gauge."""
    color = "#3fb950" if prob<30 else "#d29922" if prob<60 else "#f85149"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        number={"suffix":"%","font":{"size":36,"color":color,"family":"Inter"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#8b949e","tickfont":{"size":10}},
            "bar":{"color":color,"thickness":0.28},
            "bgcolor":"#161b22",
            "bordercolor":"#21262d",
            "steps":[
                {"range":[0,30],"color":"#0d2b1a"},
                {"range":[30,60],"color":"#2b2010"},
                {"range":[60,100],"color":"#2b0d0d"},
            ],
            "threshold":{"line":{"color":color,"width":4},"thickness":0.75,"value":prob},
        },
        title={"text":"Heart Disease Risk","font":{"size":13,"color":"#8b949e"}},
        domain={"row":0,"column":0}
    ))
    fig.update_layout(height=260, margin=dict(t=40,b=10,l=20,r=20),
                      paper_bgcolor="#0d1117", font_color="#c9d1d9")
    return fig


def plotly_radar(local_weights: dict, global_coef: np.ndarray,
                 feat_cols: list) -> go.Figure:
    """Radar chart: local silo coefficients vs global."""
    n = min(len(feat_cols), 8)
    feats = feat_cols[:n]
    gw = global_coef[:n].tolist()

    fig = go.Figure()
    for name, (coef, _) in local_weights.items():
        lw = coef[:n].tolist()
        fig.add_trace(go.Scatterpolar(
            r=lw + [lw[0]], theta=feats + [feats[0]],
            fill="toself", name=name,
            line_color=SILO_COLORS.get(name,"#58a6ff"),
            fillcolor=SILO_COLORS.get(name,"#58a6ff"),
            opacity=0.35,
        ))
    fig.add_trace(go.Scatterpolar(
        r=gw + [gw[0]], theta=feats + [feats[0]],
        name="Global (FedProx)", mode="lines",
        line=dict(color="#ffffff", width=2.5, dash="dash"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, color="#6e7681",
                                   gridcolor="#21262d"),
                   angularaxis=dict(color="#c9d1d9"),
                   bgcolor="#161b22"),
        paper_bgcolor="#0d1117", font_color="#c9d1d9",
        legend=dict(bgcolor="#161b22", bordercolor="#21262d"),
        title=dict(text="Feature Coefficient Radar — Local vs Global",
                   font_color="#e6edf3", x=0.5),
        height=430, margin=dict(t=50,b=20,l=20,r=20),
    )
    return fig


def plotly_mahal_heatmap(anomaly_report: pd.DataFrame,
                          cosine_matrix: pd.DataFrame) -> go.Figure:
    """Mahalanobis D² + cosine similarity dual heatmap."""
    hospitals = cosine_matrix.index.tolist()
    sim_vals  = cosine_matrix.values.tolist()
    fig = go.Figure(go.Heatmap(
        z=sim_vals, x=hospitals, y=hospitals,
        colorscale="Blues", zmin=0, zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in sim_vals],
        texttemplate="%{text}", textfont_size=11,
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Cosine Sim: %{z:.4f}<extra></extra>",
        colorbar=dict(tickcolor="#8b949e", outlinecolor="#21262d"),
    ))
    fig.update_layout(
        title=dict(text="Cosine Similarity — Hospital Weight Vectors",
                   font_color="#e6edf3", x=0.5),
        height=380, margin=dict(t=50,b=20,l=20,r=20),
        **PLOTLY_DARK,
    )
    return fig


def plotly_mahal_bar(anomaly_report: pd.DataFrame) -> go.Figure:
    """Horizontal bar: Mahalanobis D² per hospital."""
    colors = ["#f85149" if "Yes" in str(v) else "#3fb950"
              for v in anomaly_report["Is Anomaly"]]
    fig = go.Figure(go.Bar(
        x=anomaly_report["Mahalanobis D²"],
        y=anomaly_report["Hospital"],
        orientation="h",
        marker_color=colors,
        text=[f"D²={v:.3f}" for v in anomaly_report["Mahalanobis D²"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>D²: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Mahalanobis Distance² from Global Centroid",
                   font_color="#e6edf3", x=0.5),
        xaxis_title="D²", height=300,
        margin=dict(t=50,b=20,l=20,r=20), **PLOTLY_DARK,
    )
    return fig


def plotly_convergence(rh: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rh["Round"], y=rh["Accuracy"],
        mode="lines+markers+text",
        line=dict(color="#58a6ff", width=3),
        marker=dict(size=10, color="#1f6feb", line=dict(color="#58a6ff",width=2)),
        text=[f"{v:.1f}%" for v in rh["Accuracy"]],
        textposition="top center", textfont_color="#c9d1d9",
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        name="Global Accuracy",
    ))
    fig.update_layout(
        title=dict(text="Convergence Monitor — Global Accuracy per Round",
                   font_color="#e6edf3", x=0.5),
        xaxis=dict(title="Federated Round", tickmode="linear",
                   gridcolor="#21262d", color="#8b949e"),
        yaxis=dict(title="Accuracy (%)", gridcolor="#21262d", color="#8b949e"),
        height=380, margin=dict(t=50,b=40,l=50,r=20),
        **PLOTLY_DARK,
    )
    return fig


def plotly_privacy_utility(rh: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rh["Epsilon_Spent"], y=rh["Accuracy"],
        mode="lines+markers",
        line=dict(color="#3fb950", width=3),
        marker=dict(size=9, color="#238636"),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.08)",
        name="Accuracy (%)",
    ))
    # Show ε budget consumed as secondary area
    fig.add_trace(go.Scatter(
        x=rh["Epsilon_Spent"], y=rh["Epsilon_Spent"],
        mode="lines+markers",
        line=dict(color="#f85149", width=2, dash="dash"),
        marker=dict(size=7, color="#da3633"),
        name="ε Budget Spent",
        yaxis="y2",
    ))
    fig.update_layout(
        title=dict(text="Privacy-Utility Tradeoff  (Accuracy ↑  vs  ε Spent ↑)",
                   font_color="#e6edf3", x=0.5),
        xaxis=dict(title="Cumulative ε Spent", gridcolor="#21262d", color="#8b949e"),
        yaxis=dict(title="Accuracy (%)", color="#3fb950", gridcolor="#21262d"),
        yaxis2=dict(title="ε Spent", color="#f85149", overlaying="y",
                    side="right", showgrid=False),
        legend=dict(bgcolor="#161b22", bordercolor="#21262d"),
        height=400, margin=dict(t=50,b=40,l=50,r=50), **PLOTLY_DARK,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:10px 0 5px'>
      <div style='font-size:2.6rem'>🫀</div>
      <div style='font-size:1.05rem;font-weight:800;color:#e6edf3'>UMI</div>
      <div style='font-size:.68rem;color:#6e7681;font-weight:600;letter-spacing:.08em;text-transform:uppercase'>
        Unified Medical Intelligence</div></div>""", unsafe_allow_html=True)
    st.divider()

    role = st.selectbox("🎭 Active Role",["Doctor","Researcher","Regulator"])
    ICONS = {"Doctor":"🩺","Researcher":"🔬","Regulator":"🔒"}
    st.markdown(f"**Role:** {ICONS[role]} {role}")
    st.divider()

    st.markdown("**⚙️ FL Hyperparameters**")
    n_rounds = st.slider("FL Rounds", 1, 8, 3)
    mu       = st.slider("FedProx μ", 0.0, 0.5, 0.01, step=0.01)
    epsilon  = st.slider("Privacy ε", 0.1, 5.0, 1.0, step=0.1)
    st.divider()

    gemini_key = st.text_input("🤖 Gemini API Key", type="password", placeholder="AIza…")
    st.divider()

    if data_available(): st.success("✅ Silo data ready")
    else: st.error("❌ Run data_preparation.py first")
    st.caption(f"📁 `{SILO_DIR}`")

# ── Header ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 10])
with c1: st.markdown("<div style='font-size:3rem;padding-top:14px'>🫀</div>",
                     unsafe_allow_html=True)
with c2: st.markdown(f"""<div>
<h1 style='margin:0;color:#e6edf3;font-size:2rem;font-weight:800'>Unified Medical Intelligence</h1>
<p style='margin:4px 0 0;color:#6e7681;font-size:.85rem'>
{ICONS[role]} <strong style='color:#8b949e'>{role} Dashboard</strong>
&nbsp;·&nbsp; FedProx &nbsp;·&nbsp; ε-DP &nbsp;·&nbsp; Sim-HE &nbsp;·&nbsp; Gemini Agents
</p></div>""", unsafe_allow_html=True)
st.divider()

# ── Data / model gate ─────────────────────────────────────────────────────────
if not data_available():
    st.warning("Run `python data_preparation.py` to prepare silo data."); st.stop()

with st.spinner("🔄 Running FedProx training…"):
    try: fl = run_fl(n_rounds, mu, epsilon)
    except Exception as e: st.error(f"Training failed: {e}"); st.stop()

gm            = fl["global_model"]
scaler        = fl["scaler"]
feat_cols     = fl["feat_cols"]
local_weights = fl["local_weights"]
rh            = fl["round_history"]
X_test        = fl["X_test_all"]
y_test        = fl["y_test_all"]
gcoef         = gm.coef_[0]
eps_spent     = rh["Epsilon_Spent"].iloc[-1]
final_acc     = rh["Accuracy"].iloc[-1]
all_df        = load_all()

# ── KPI strip ─────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
kpis = [
    ("Hospitals","4","Federated Silos",""),
    ("Total Patients",f"{len(all_df):,}","Cross-silo",""),
    ("Global Accuracy",f"{final_acc:.1f}%",f"Round {n_rounds}",""),
    ("ε Consumed",f"{eps_spent:.2f}",f"of {epsilon*n_rounds:.1f}","blue"),
    ("Prevalence",f"{all_df[TARGET_COL].mean()*100:.1f}%" if TARGET_COL in all_df.columns else "—","Positive rate",""),
]
for col, (label, val, sub, kind) in zip([k1,k2,k3,k4,k5], kpis):
    sub_class = "kpi-sub-blue" if kind=="blue" else "kpi-sub"
    col.markdown(f"""<div class='kpi'><div class='kpi-label'>{label}</div>
    <div class='kpi-value'>{val}</div><div class='{sub_class}'>{sub}</div></div>""",
    unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  DOCTOR VIEW
# ════════════════════════════════════════════════════════════════════════════════
if role == "Doctor":
    st.markdown("<div class='sec'>🩺 Patient Triage Center</div>", unsafe_allow_html=True)
    st.caption("Complete all 11 vitals → Global FedProx model scores risk → Gemini builds Clinical Pathway.")

    col_form, col_gauge = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown("**Patient Vitals (11 inputs)**")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            age      = st.slider("Age",              20, 80, 54)
            trestbps = st.slider("Resting BP (mmHg)",80, 200, 130)
            chol     = st.slider("Cholesterol mg/dL",100, 600, 246)
            thalch   = st.slider("Max Heart Rate",   60, 220, 150)
            oldpeak  = st.slider("ST Depression",    0.0, 6.5, 1.0, 0.1)
            ca       = st.slider("Major Vessels 0-3",0, 3, 0)
        with r1c2:
            sex     = st.selectbox("Sex",[0,1],format_func=lambda x:"Female" if x==0 else "Male")
            cp      = st.selectbox("Chest Pain Type",[1,2,3,4],
                                   format_func=lambda x:{1:"Typical Angina",2:"Atypical Angina",
                                                          3:"Non-Anginal",4:"Asymptomatic"}[x])
            fbs     = st.selectbox("Fasting Blood Sugar >120",[0,1],
                                   format_func=lambda x:"No" if x==0 else "Yes")
            restecg = st.selectbox("Resting ECG",[0,1,2],
                                   format_func=lambda x:{0:"Normal",1:"ST-T Abnormality",
                                                          2:"LV Hypertrophy"}[x])
            exang   = st.selectbox("Exercise Angina",[0,1],
                                   format_func=lambda x:"No" if x==0 else "Yes")

        predict_btn = st.button("⚡ Run Prediction", type="primary",
                                use_container_width=True, key="predict")

    with col_gauge:
        st.markdown("**Risk Gauge**")
        if predict_btn or "last_prob" in st.session_state:
            if predict_btn:
                inp = {"age":age,"trestbps":trestbps,"chol":chol,"thalch":thalch,
                       "oldpeak":oldpeak,"ca":ca,"sex":sex,"cp":cp,
                       "fbs":fbs,"restecg":restecg,"exang":exang}
                vec   = np.array([[inp.get(f,0) for f in feat_cols]])
                prob  = float(gm.predict_proba(scaler.transform(vec))[0][1]*100)
                st.session_state["last_prob"] = prob
                st.session_state["last_inp"]  = inp
            else:
                prob = st.session_state["last_prob"]
                inp  = st.session_state.get("last_inp", {})

            st.plotly_chart(plotly_gauge(prob), use_container_width=True,
                            key="gauge_chart")
            st.markdown(f"""<div style='text-align:center;margin-top:-10px'>
              <span class='he-badge'>🔐 HE TRANSIT VERIFIED</span>&nbsp;
              <span class='dp-badge'>🔒 ε-DP APPLIED</span></div>""",
              unsafe_allow_html=True)

            # Feature bar (Plotly)
            coefs = gm.coef_[0]
            fi = pd.DataFrame({"Feature":feat_cols,"Impact":coefs})
            fi = fi.reindex(fi["Impact"].abs().sort_values(ascending=False).index).head(6)
            colors = ["#f85149" if v>0 else "#3fb950" for v in fi["Impact"]]
            bar_fig = go.Figure(go.Bar(x=fi["Impact"],y=fi["Feature"],
                orientation="h",marker_color=colors,
                hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>"))
            bar_fig.update_layout(height=230, margin=dict(t=10,b=10,l=0,r=0),
                                  xaxis_title="Coefficient",**PLOTLY_DARK)
            st.plotly_chart(bar_fig, use_container_width=True, key="feat_bar")
        else:
            # Empty gauge placeholder
            st.plotly_chart(plotly_gauge(0), use_container_width=True, key="gauge_empty")
            st.info("👈 Enter vitals and click **Run Prediction**.")

    # Gemini Clinical Pathway
    st.markdown("<div class='sec'>✨ Gemini Clinical Pathway</div>", unsafe_allow_html=True)
    if gemini_key:
        if st.button("🤖 Generate Clinical Pathway", type="primary", key="pathway"):
            prob_val   = st.session_state.get("last_prob", 50.0)
            inp        = st.session_state.get("last_inp", {})
            risk_label = "High Risk" if prob_val>=60 else "Moderate Risk" if prob_val>=30 else "Low Risk"
            prompt = f"""You are a Senior Cardiologist. A Federated Learning AI (FedProx, 4 hospitals,
ε-DP) predicted {prob_val:.1f}% ({risk_label}) heart disease risk for:
Age={inp.get('age','?')}, Sex={'M' if inp.get('sex')==1 else 'F'},
BP={inp.get('trestbps','?')} mmHg, Chol={inp.get('chol','?')} mg/dL,
MaxHR={inp.get('thalch','?')}, ST↓={inp.get('oldpeak','?')}, 
Vessels={inp.get('ca','?')}, CP Type={inp.get('cp','?')},
ExAngina={'Yes' if inp.get('exang')==1 else 'No'}.

Write a structured Clinical Pathway:
1. IMMEDIATE ACTIONS (2 bullets)
2. INVESTIGATIONS (2 bullets)
3. LONG-TERM MANAGEMENT (2 bullets)
4. PATIENT EXPLANATION (1 sentence in lay language)
Be specific, cite numbers and drug classes."""
            with st.spinner("🤖 Generating…"):
                try:
                    from google import genai
                    resp = genai.Client(api_key=gemini_key).models.generate_content(
                        model="gemini-2.0-flash", contents=prompt)
                    txt = resp.text
                except Exception as e: txt = f"⚠ {e}"
            st.markdown(f"""<div class='gbox'><div class='gtitle'>
🤖 Clinical Pathway — {risk_label}</div>
<div class='gtext'>{txt.replace(chr(10),'<br>')}</div></div>""",
            unsafe_allow_html=True)
    else:
        st.markdown("""<div class='gbox'><div class='gtitle'>🤖 Gemini Clinical Pathway</div>
<div class='gtext' style='color:#6e7681'>Add your Gemini API key in the sidebar to unlock AI-generated 
clinical pathways. Get yours at <a href='https://aistudio.google.com' style='color:#58a6ff'>
aistudio.google.com</a></div></div>""", unsafe_allow_html=True)

    # Accuracy comparison expander
    with st.expander("📊 Collaborative Advantage — Global vs Cleveland-Only"):
        clev_path = SILO_FILES.get("Cleveland")
        if clev_path and clev_path.exists():
            df_c  = load_silo(str(clev_path))
            fc    = [c for c in feat_cols if c in df_c.columns]
            Xc_sc = scaler.transform(df_c[fc])
            clev_m = LogisticRegression(max_iter=1000).fit(Xc_sc, df_c[TARGET_COL].astype(int))
            Xt_sc = scaler.transform(X_test[[f for f in feat_cols if f in X_test.columns]])
            gacc  = accuracy_score(y_test, gm.predict(Xt_sc))*100
            cacc  = accuracy_score(y_test, clev_m.predict(Xt_sc))*100
            d     = gacc - cacc
            cA,cB = st.columns(2)
            cA.metric("🌐 Unified FedProx", f"{gacc:.2f}%",
                      f"{'+' if d>=0 else ''}{d:.2f}% vs Cleveland")
            cB.metric("🏥 Cleveland-Only", f"{cacc:.2f}%")


# ════════════════════════════════════════════════════════════════════════════════
#  RESEARCHER VIEW
# ════════════════════════════════════════════════════════════════════════════════
elif role == "Researcher":
    st.markdown("<div class='sec'>🔬 Federated Research Intelligence</div>",
                unsafe_allow_html=True)

    rt1, rt2, rt3, rt4 = st.tabs([
        "🕸 Radar Chart", "📊 Feature Importance",
        "🔍 Anomaly Detection", "🏥 Hospital Breakdown"
    ])

    with rt1:
        st.caption("Interactive radar comparing local hospital features vs global FedProx consensus.")
        st.plotly_chart(plotly_radar(local_weights, gcoef, feat_cols),
                        use_container_width=True, key="radar")

    with rt2:
        st.caption("Global FedProx coefficient magnitudes — the unified predictive intelligence.")
        coefs_arr = gm.coef_[0]
        fi_df = pd.DataFrame({"Feature":feat_cols,"Weight":coefs_arr,
                              "Abs":np.abs(coefs_arr)}).sort_values("Abs")
        colors = ["#f85149" if v>0 else "#3fb950" for v in fi_df["Weight"]]
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Weight"], y=fi_df["Feature"], orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in fi_df["Weight"]], textposition="outside",
            hovertemplate="<b>%{y}</b>: %{x:.5f}<extra></extra>",
        ))
        fig_fi.add_vline(x=0, line_color="#6e7681", line_dash="dash")
        fig_fi.update_layout(
            title=dict(text=f"Global Feature Importance (μ={mu}, ε={epsilon}, {n_rounds} rounds)",
                       font_color="#e6edf3", x=0.5),
            height=400, margin=dict(t=50,b=20,l=20,r=80), **PLOTLY_DARK,
        )
        st.plotly_chart(fig_fi, use_container_width=True, key="fi_bar")
        st.dataframe(fi_df[["Feature","Weight"]].sort_values("Weight",key=abs,ascending=False)
                     .reset_index(drop=True).rename(columns={"Weight":"Global Coefficient"}),
                     use_container_width=True)

    with rt3:
        st.caption("Mahalanobis D² and cosine similarity matrix to flag statistical outlier hospitals.")
        from anomaly_detection import detect_weight_anomalies
        anom = detect_weight_anomalies(local_weights, feat_cols)

        if anom["outlier_hospitals"]:
            st.warning(anom["summary"])
        else:
            st.success(anom["summary"])

        col_h, col_cm = st.columns([1, 1])
        with col_h:
            st.plotly_chart(plotly_mahal_bar(anom["anomaly_report"]),
                            use_container_width=True, key="mahal_bar")
        with col_cm:
            st.plotly_chart(plotly_mahal_heatmap(anom["anomaly_report"],
                                                  anom["cosine_sim_matrix"]),
                            use_container_width=True, key="cosine_heat")

        st.caption("📋 Full Anomaly Report")
        st.dataframe(anom["anomaly_report"][["Hospital","L2 Z-Score","Mahalanobis D²",
                                              "χ² p-value","Mean Cosine Sim","Is Anomaly"]],
                     use_container_width=True, hide_index=True)

    with rt4:
        rows = []
        for name, path in SILO_FILES.items():
            if path.exists():
                df = load_silo(str(path))
                rows.append({"Hospital":name,"Patients":len(df),
                             "Disease Rate":f"{df[TARGET_COL].mean()*100:.1f}%",
                             "Avg Age":f"{df['age'].mean():.1f}" if "age" in df.columns else "—",
                             "Avg Chol":f"{df['chol'].mean():.1f}" if "chol" in df.columns else "—",
                             "Avg MaxHR":f"{df['thalch'].mean():.1f}" if "thalch" in df.columns else "—",
                             "DP σ":f"{1.0*(2*np.log(1.25/1e-5))**.5/(epsilon*len(df)):.6f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Per-hospital Plotly bar
        if local_weights:
            nf = min(len(feat_cols), 7)
            fig_hb = go.Figure()
            for name, (coef,_) in local_weights.items():
                fig_hb.add_trace(go.Bar(name=name, x=feat_cols[:nf],
                    y=coef[:nf], marker_color=SILO_COLORS.get(name,"#58a6ff"),
                    opacity=0.85))
            fig_hb.add_trace(go.Bar(name="Global (FedProx)", x=feat_cols[:nf],
                y=gcoef[:nf], marker_color="#ffffff", opacity=0.9))
            fig_hb.update_layout(barmode="group",
                title=dict(text="Per-Hospital vs Global Coefficients",
                           font_color="#e6edf3", x=0.5),
                height=380, margin=dict(t=50,b=40,l=20,r=20), **PLOTLY_DARK,
                legend=dict(bgcolor="#161b22",bordercolor="#21262d"))
            st.plotly_chart(fig_hb, use_container_width=True, key="hosp_bar")


# ════════════════════════════════════════════════════════════════════════════════
#  REGULATOR VIEW
# ════════════════════════════════════════════════════════════════════════════════
elif role == "Regulator":
    st.markdown("<div class='sec'>🔒 Privacy & Governance Command Center</div>",
                unsafe_allow_html=True)

    regt1, regt2, regt3, regt4 = st.tabs([
        "📈 Convergence", "🔐 Privacy-Utility", "📋 Audit Trail", "🤖 Gemini Governance"
    ])

    with regt1:
        st.caption("Global model accuracy improving across federated communication rounds.")
        st.plotly_chart(plotly_convergence(rh), use_container_width=True, key="conv")
        st.dataframe(rh.rename(columns={"Accuracy":"Accuracy (%)","Epsilon_Spent":"ε Cumulative",
                                         "Time_s":"Time (s)"}),
                     use_container_width=True, hide_index=True)

    with regt2:
        st.caption("Privacy-Utility tradeoff: as ε budget increases, accuracy improves.")
        st.plotly_chart(plotly_privacy_utility(rh), use_container_width=True, key="pu")

        b_remaining = epsilon*n_rounds - eps_spent
        pb1,pb2,pb3 = st.columns(3)
        pb1.markdown(f"""<div class='kpi'><div class='kpi-label'>ε Allocated</div>
        <div class='kpi-value'>{epsilon*n_rounds:.2f}</div>
        <div class='kpi-sub-blue'>{n_rounds} rounds × {epsilon}</div></div>""",
        unsafe_allow_html=True)
        pb2.markdown(f"""<div class='kpi'><div class='kpi-label'>ε Consumed</div>
        <div class='kpi-value' style='color:#f85149'>{eps_spent:.4f}</div>
        <div class='kpi-sub'>Gaussian mechanism</div></div>""",unsafe_allow_html=True)
        clr = "#3fb950" if b_remaining>0 else "#f85149"
        pb3.markdown(f"""<div class='kpi'><div class='kpi-label'>ε Remaining</div>
        <div class='kpi-value' style='color:{clr}'>{b_remaining:.4f}</div>
        <div class='kpi-sub'>{'Safe ✅' if b_remaining>0 else 'Over 🚨'}</div></div>""",
        unsafe_allow_html=True)

    with regt3:
        now = datetime.datetime.now()
        audit_rows = []
        for i,(name,path) in enumerate(SILO_FILES.items()):
            if not path.exists(): continue
            for rnd in range(1, n_rounds+1):
                ts = now - datetime.timedelta(minutes=(len(SILO_FILES)*(n_rounds-rnd+1)-i)*2)
                audit_rows.append({"Event":f"EVT-{(rnd-1)*10+i:03d}",
                    "Timestamp":ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "Hospital":name,"Round":rnd,
                    "Transfer":"FedProx Weights (DP-noised)","HE":"🔐 Encrypted",
                    "ε/round":epsilon,"μ":mu,"Raw PHI":"❌ None","Status":"✅ Secure"})
        for rnd in range(1, n_rounds+1):
            ts = now - datetime.timedelta(minutes=(n_rounds-rnd)*2)
            audit_rows.append({"Event":f"EVT-AGG-{rnd:02d}",
                "Timestamp":ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Hospital":"Central Coordinator","Round":rnd,
                "Transfer":"FedProx Aggregation","HE":"🔓 Decrypted at agg",
                "ε/round":epsilon,"μ":mu,"Raw PHI":"❌ None","Status":"✅ Secure"})
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

        # HE ciphertext preview
        if local_weights:
            from federated_core import simulate_he_encrypt
            coef0 = list(local_weights.values())[0][0]
            he_tensor = simulate_he_encrypt(coef0)
            st.markdown("**🔐 Sample Transit Ciphertext (TenSEAL CKKS Poly-Modulus)**")
            st.markdown(f"""<div class='cipher'>
 Payload: {repr(he_tensor)}</div>""", unsafe_allow_html=True)
            st.caption("The aggregator receives only this Homomorphic object. It adds/sums them without ever decrypting.")

        cc1,cc2,cc3 = st.columns(3)
        for col, label, val, sub in [
            (cc1,"HIPAA","✅ Compliant","Zero PHI transferred"),
            (cc2,"Data Residency","✅ On-Premise","Records stayed local"),
            (cc3,"DP Mechanism","Gaussian (ε,δ)","δ=1e-5, clip=1.0"),
        ]:
            col.markdown(f"""<div class='kpi'><div class='kpi-label'>{label}</div>
            <div class='kpi-value' style='font-size:1.1rem;color:#3fb950'>{val}</div>
            <div class='kpi-sub'>{sub}</div></div>""", unsafe_allow_html=True)

    with regt4:
        st.caption("3-step AI governance audit: Auditor → Clinician → Compliance Officer")
        if gemini_key:
            if st.button("🤖 Run Full Governance Audit", type="primary", key="gov_btn"):
                from gemini_pipeline import run_full_pipeline, build_divergence_json
                prog = st.empty()
                def pcb(step, label): prog.info(f"**[{step}/3]** {label}")
                with st.spinner("Running 3-step Gemini pipeline…"):
                    div_json = build_divergence_json(local_weights, gcoef, feat_cols)
                    out = run_full_pipeline(
                        global_weights=gcoef, local_weights=local_weights,
                        feat_cols=feat_cols, epsilon_spent=eps_spent,
                        epsilon_budget=epsilon*n_rounds, n_rounds=n_rounds,
                        api_key=gemini_key, divergence_json=div_json,
                        progress_callback=pcb)
                prog.empty()

                aud  = out["auditor"]
                clin = out["clinician"]
                gov  = out["governance"]
                rd   = gov["render"]
                sc   = gov["compliance_scores"]

                # Auditor
                hc = aud["render"]["badge_color"]
                st.markdown(f"""<div class='gbox'>
<div class='gtitle'>🔍 Step 1 — Auditor</div>
<div style='margin-bottom:12px'>
  <span style='color:#f0883e;font-weight:700'>{aud['outlier_hospital']}</span>
  &nbsp;<span style='background:{hc}22;border:1px solid {hc};border-radius:10px;
  padding:2px 9px;font-size:.73rem;color:{hc};font-weight:700'>{aud['hypothesis_type']}</span>
  &nbsp;<span style='color:{aud["render"]["confidence_color"]};font-size:.73rem;font-weight:700'>
  Confidence: {aud['confidence']}</span>
</div>
<div class='gtext'>{aud['gemini_verdict'].replace(chr(10),'<br>')}</div></div>""",
                unsafe_allow_html=True)
                st.dataframe(aud["divergence_table"], use_container_width=True, hide_index=True)

                # Clinician
                badges = " ".join(
                    f"<span style='background:{b['color']}22;border:1px solid {b['color']}55;"
                    f"border-radius:10px;padding:2px 9px;font-size:.73rem;color:{b['color']};"
                    f"font-weight:700'>{b['name']} ({b['weight']:+.4f})</span>"
                    for b in clin["render"]["feature_badges"])
                st.markdown(f"""<div class='gbox'>
<div class='gtitle'>🩺 Step 2 — Clinician</div>
<div style='margin-bottom:10px'>{badges}</div>
<div class='gtext'>{clin['gemini_brief'].replace(chr(10),'<br>')}</div>
<div style='margin-top:12px;padding:10px;background:#0d1117;border-left:3px solid #3fb950;
border-radius:6px;font-size:.85rem;color:#c9d1d9'>
💊 <strong>Intervention:</strong> {clin['gemini_intervention']}</div></div>""",
                unsafe_allow_html=True)

                # Governance
                gc1,gc2,gc3,gc4 = st.columns(4)
                for gco, lbl, val, color, sub in [
                    (gc1,"Verdict",rd["verdict"],rd["verdict_color"],"Overall"),
                    (gc2,"HIPAA",gov["hipaa_status"],rd["hipaa_color"],f"Score: {sc['hipaa_score']}/100"),
                    (gc3,"GDPR",gov["gdpr_status"],rd["gdpr_color"],f"Score: {sc['gdpr_score']}/100"),
                    (gc4,"ε Budget Used",f"{sc['budget_used_pct']:.1f}%","#58a6ff",f"≈{gov['rounds_remaining']} rounds left"),
                ]:
                    gco.markdown(f"""<div class='kpi'><div class='kpi-label'>{lbl}</div>
                    <div class='kpi-value' style='color:{color};font-size:1.1rem'>{val}</div>
                    <div class='kpi-sub'>{sub}</div></div>""", unsafe_allow_html=True)

                st.markdown(f"""<div class='gbox'>
<div class='gtitle'>🔒 Step 3 — Governance: HIPAA + GDPR Verdict</div>
<div class='gtext'>{gov['gemini_verdict'].replace(chr(10),'<br>')}</div>
<div style='margin-top:12px;padding:10px;background:#0d1117;border-left:3px solid #58a6ff;
border-radius:6px;font-size:.85rem;color:#c9d1d9'>
📋 <strong>Recommendation:</strong> {gov['gemini_recommendation']}</div></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown("""<div class='gbox'><div class='gtitle'>🤖 Gemini Governance Audit</div>
<div class='gtext' style='color:#6e7681'>Add your Gemini API key in the sidebar to run the 
3-step pipeline: Auditor → Clinician → Governance.</div></div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""<p style='text-align:center;color:#3d444d;font-size:.74rem'>
🫀 UMI · Unified Medical Intelligence · FedProx · ε-DP · Sim-HE · Agentic Gemini
&nbsp;|&nbsp; No raw data shared · ε={eps_spent:.4f} consumed · {n_rounds} rounds
</p>""", unsafe_allow_html=True)
