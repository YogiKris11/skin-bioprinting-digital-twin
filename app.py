from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
SOURCE_PATH = DATA / "cect_cleaned_807_source_records.csv"
SCENARIO_PATH = DATA / "derived_5000_parameter_scenarios_NOT_EXPERIMENTAL.csv"
FIT_PATH = DATA / "ml_model_fit_inputs_4400.csv"
INTEGRATED_PATH = DATA / "integrated_dataset_explorer_5807.csv"
MODEL_PATH = ROOT / "models" / "cect_isolation_forest.joblib"
FEATURES = ["pressure_min_kpa", "speed_min_mm_s", "temperature_min_c", "needle_diameter_um_explicit"]
FEATURE_LABELS = {
    "pressure_min_kpa": "Pressure (kPa)",
    "speed_min_mm_s": "Printing speed (mm/s)",
    "temperature_min_c": "Temperature (°C)",
    "needle_diameter_um_explicit": "Nozzle diameter (µm)",
}

st.set_page_config(page_title="Skin Bioprinting Digital Twin", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at 12% 0%, #142c43 0%, #0a1422 42%, #08111d 100%); color:#e8f0f8; }
.block-container {max-width: 1580px; padding-top: 1.35rem; padding-bottom: 2.5rem;}
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
h1,h2,h3 {font-family:'Manrope',sans-serif; letter-spacing:-.035em;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#101e30,#0b1523); border-right:1px solid #20354b;}
[data-testid="stMetric"] {background:linear-gradient(145deg,rgba(24,48,72,.92),rgba(14,29,46,.92)); border:1px solid #29445e; border-radius:16px; padding:17px 19px; box-shadow:0 8px 24px rgba(0,0,0,.13);}
[data-testid="stMetricLabel"] {color:#a9bfd2!important; font-size:.82rem;}
[data-testid="stMetricValue"] {color:#f4f8fc!important; font-family:'Manrope',sans-serif;}
[data-testid="stTabs"] button {font-weight:650; color:#9eb4c8; padding-top:13px; padding-bottom:13px;}
[data-testid="stTabs"] button[aria-selected="true"] {color:#75d5d0!important; border-bottom-color:#55c8c0!important;}
div.stButton > button, div.stDownloadButton > button {border-radius:11px; border:1px solid #3b657e; font-weight:650; min-height:42px;}
div.stButton > button[kind="primary"] {background:linear-gradient(100deg,#167e86,#2a9d9b); border:0; color:white;}
[data-testid="stDataFrame"] {border:1px solid #263e55; border-radius:12px; overflow:hidden;}
.hero {background:linear-gradient(112deg,rgba(20,49,75,.98),rgba(17,86,94,.94)); border:1px solid #2a6470; border-radius:22px; padding:25px 30px; margin-bottom:20px; box-shadow:0 18px 50px rgba(0,0,0,.18);}
.hero-kicker {font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:#86ded6; margin-bottom:8px;}
.hero-title {font-family:'Manrope',sans-serif; font-size:2.05rem; font-weight:800; color:#f5fbff; line-height:1.18;}
.hero-sub {color:#c4dce7; margin-top:8px; font-size:.98rem;}
.section-note {color:#94aabd; font-size:.86rem;}
.panel {background:rgba(16,32,49,.72); border:1px solid #233b52; border-radius:16px; padding:16px 18px;}
.status-pill {display:inline-block; padding:5px 10px; border-radius:99px; background:#153d48; color:#91e4dc; font-size:.75rem; font-weight:700; letter-spacing:.03em;}
hr {border-color:#263b50!important;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading process datasets…")
def load_data():
    source = pd.read_csv(SOURCE_PATH)
    scenarios = pd.read_csv(SCENARIO_PATH)
    fit = pd.read_csv(FIT_PATH)
    integrated = pd.read_csv(INTEGRATED_PATH)
    return source, scenarios, fit, integrated

@st.cache_resource(show_spinner="Loading trained screening model…")
def load_model():
    return joblib.load(MODEL_PATH)

required = [SOURCE_PATH, SCENARIO_PATH, FIT_PATH, INTEGRATED_PATH, MODEL_PATH]
missing = [p.name for p in required if not p.exists()]
if missing:
    st.error("The application package is incomplete. Missing: " + ", ".join(missing))
    st.stop()
try:
    source_df, scenarios_df, fit_raw, integrated_df = load_data()
    model = load_model()
except Exception as exc:
    st.error(f"Unable to initialise the application: {exc}")
    st.stop()

# The explorer is built from the complete 4-feature model input file, so every row has
# the four numeric columns needed for filtering, plotting and export.
fit_df = fit_raw.copy()
fit_df["record_id"] = [f"PROC-{i:05d}" for i in range(1, len(fit_df) + 1)]
fit_df["record_basis"] = fit_df["input_group"].map({
    "Published source parameter record": "Published source parameter record",
    "Scenario-derived parameter sample": "Scenario-derived parameter sample",
}).fillna(fit_df["input_group"])
fit_df["source_context"] = fit_df["input_group"].map({
    "Published source parameter record": "CECT open-source bioink database",
    "Scenario-derived parameter sample": "CECT-linked parameter-space scenario",
}).fillna("Reference dataset")

st.markdown("""
<div class="hero">
  <div class="hero-kicker">FYP 2026–27 · Intelligent process analytics</div>
  <div class="hero-title">🧬 Skin Bioprinting Digital Twin</div>
  <div class="hero-sub">A unified workspace for process-data exploration, operating-window analysis and machine-learning-based anomaly screening.</div>
</div>
""", unsafe_allow_html=True)

# Workspace controls are defined once and feed every downstream view.
with st.sidebar:
    st.markdown("## Digital Twin")
    st.markdown('<span class="status-pill">SYSTEM ONLINE</span>', unsafe_allow_html=True)
    st.caption("Process intelligence · Live workspace")
    st.divider()
    st.markdown("**Navigation**")
    st.caption("Use the Workspace tab to configure process parameters and dataset filters.")
    st.divider()
    st.caption("FYP2026-27-013 · Prototype build")

tabs = st.tabs(["Workspace", "Overview", "ML Process Screening", "Dataset Explorer", "Parameter Lab", "Methodology & Sources"])

with tabs[0]:
    st.subheader("Process Control Workspace")
    st.markdown("<div class='section-note'>Configure an operating point, screen it against the trained model, and filter the records used throughout the application.</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("#### Operating point")
    a, b, c, d = st.columns(4)
    pressure = a.number_input("Extrusion pressure (kPa)", min_value=0.0, max_value=2000.0, value=100.0, step=5.0, key="ws_pressure")
    speed = b.number_input("Printing speed (mm/s)", min_value=0.0, max_value=200.0, value=5.0, step=0.5, key="ws_speed")
    temp = c.number_input("Temperature (°C)", min_value=-20.0, max_value=250.0, value=25.0, step=1.0, key="ws_temp")
    nozzle = d.number_input("Nozzle diameter (µm)", min_value=0.0, max_value=2500.0, value=300.0, step=10.0, key="ws_nozzle")

    w1, w2, w3 = st.columns([1, 1, 1])
    if w1.button("Run live screening", type="primary", use_container_width=True, key="ws_screen"):
        point = pd.DataFrame([[pressure, speed, temp, nozzle]], columns=FEATURES)
        try:
            st.session_state["workspace_screening"] = {
                "prediction": int(model.predict(point)[0]),
                "decision": float(model.decision_function(point)[0]),
                "anomaly": float(model.score_samples(point)[0]),
                "point": [pressure, speed, temp, nozzle],
            }
        except Exception as exc:
            st.error(f"Screening failed: {exc}")
    if w2.button("Save configuration", use_container_width=True, key="ws_save"):
        saved = st.session_state.get("saved_process_configs", [])
        saved.append({"name": f"Configuration {len(saved)+1}", "pressure": pressure, "speed": speed, "temperature": temp, "nozzle": nozzle})
        st.session_state["saved_process_configs"] = saved
    if w3.button("Reset filters", use_container_width=True, key="ws_reset"):
        for key in ["ws_basis", "ws_search", "ws_pressure_range", "ws_speed_range", "ws_temp_range", "ws_nozzle_range"]:
            st.session_state.pop(key, None)
        st.rerun()

    screen = st.session_state.get("workspace_screening")
    if screen:
        s1, s2, s3 = st.columns(3)
        if screen["prediction"] == 1:
            s1.success("Within learned reference pattern")
        else:
            s1.warning("Unusual relative to learned reference pattern")
        s2.metric("Decision function", f"{screen['decision']:.5f}")
        s3.metric("Anomaly score", f"{screen['anomaly']:.5f}")
        st.caption("The screening result indicates statistical unusualness relative to the model-fit distribution; it is not a print-quality probability.")

    saved_configs = st.session_state.get("saved_process_configs", [])
    if saved_configs:
        st.markdown("#### Saved configurations")
        cfg_df = pd.DataFrame(saved_configs)
        st.dataframe(cfg_df, use_container_width=True, hide_index=True)
        st.download_button("Export configurations (CSV)", cfg_df.to_csv(index=False).encode("utf-8"),
                           "saved_process_configurations.csv", "text/csv", key="ws_export_configs")

    st.divider()
    st.markdown("#### Process data")
    st.markdown("<div class='section-note'>The complete parameter dataset is available in Dataset Explorer, with all model-input records ready to inspect and export.</div>", unsafe_allow_html=True)
    data_a, data_b, data_c = st.columns(3)
    data_a.metric("Available parameter records", f"{len(fit_df):,}")
    data_b.metric("Process parameters", "04")
    data_c.metric("Data fields", "06")

# Dataset Explorer uses the complete dataset directly; it is not dependent on Workspace filters.
filtered = fit_df.copy()

# Top-level summary
k1, k2, k3, k4 = st.columns(4)
k1.metric("Model-fit parameter records", f"{len(fit_df):,}", help="Complete four-parameter rows supplied to the current anomaly-screening workflow.")
k2.metric("Integrated dataset entries", f"{len(integrated_df):,}")
k3.metric("Parameter-space scenarios", f"{len(scenarios_df):,}")
k4.metric("Model input dimensions", "04", help="Pressure, speed, temperature and nozzle diameter")
st.markdown("<div class='section-note'>Four-dimensional process monitoring · Interactive data exploration · Unsupervised anomaly screening</div>", unsafe_allow_html=True)
st.write("")

with tabs[1]:
    st.subheader("Process intelligence overview")
    left, right = st.columns([1.2, 1])
    with left:
        coverage = pd.DataFrame({"Parameter": ["Pressure", "Printing speed", "Temperature", "Nozzle diameter"],
            "Available records": [source_df[c].notna().sum() for c in FEATURES]})
        fig = px.bar(coverage, x="Parameter", y="Available records", color="Parameter", text="Available records",
                     title="Parameter availability in source dataset", template="plotly_dark")
        fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=55,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        basis_counts = fit_df["record_basis"].value_counts().rename_axis("Record basis").reset_index(name="Records")
        fig = px.pie(basis_counts, names="Record basis", values="Records", hole=.66,
                     title="Model-fit record composition", template="plotly_dark")
        fig.update_layout(margin=dict(l=10,r=10,t=55,b=10), paper_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Operating parameter distributions")
    c1, c2 = st.columns(2)
    for col, container, bins in [(FEATURES[0], c1, 35), (FEATURES[1], c2, 35)]:
        with container:
            fig = px.histogram(fit_df, x=col, nbins=bins, title=f"{FEATURE_LABELS[col]} distribution", template="plotly_dark")
            fig.update_layout(margin=dict(l=8,r=8,t=48,b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    c3, c4 = st.columns(2)
    for col, container in [(FEATURES[2], c3), (FEATURES[3], c4)]:
        with container:
            fig = px.histogram(fit_df, x=col, nbins=35, title=f"{FEATURE_LABELS[col]} distribution", template="plotly_dark")
            fig.update_layout(margin=dict(l=8,r=8,t=48,b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    st.subheader("Multivariate process space")
    fig = px.scatter(fit_df, x=FEATURES[0], y=FEATURES[1], color=FEATURES[2], size=FEATURES[3],
                     hover_data=["record_id", "record_basis"], title="Pressure vs printing speed · temperature colour · nozzle size", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Machine-learning process screening")
    st.markdown("<div class='section-note'>Evaluate an operating point against the multivariate parameter distribution learned by the Isolation Forest model.</div>", unsafe_allow_html=True)
    st.write("")
    st.info("The active operating point is controlled from the Workspace tab. Return there to adjust parameters and run a new screening.")
    workspace_result = st.session_state.get("workspace_screening")
    result = None
    if workspace_result:
        result = {"prediction": workspace_result["prediction"], "decision": workspace_result["decision"],
                  "anomaly": workspace_result["anomaly"],
                  "point": dict(zip(["Pressure", "Speed", "Temperature", "Nozzle"], workspace_result["point"]))}
    if result:
        st.divider()
        r1, r2, r3 = st.columns(3)
        if result["prediction"] == 1:
            r1.success("Within learned reference pattern")
        else:
            r1.warning("Unusual relative to learned reference pattern")
        r2.metric("Decision function", f"{result['decision']:.5f}")
        r3.metric("Anomaly score", f"{result['anomaly']:.5f}")
        st.caption("Anomaly screening indicates statistical unusualness relative to the model-fit parameter distribution. It is not a print-quality or defect probability.")
        st.dataframe(pd.DataFrame([result["point"]]), use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("**Model configuration**")
    m1,m2,m3 = st.columns(3)
    m1.metric("Algorithm", "Isolation Forest")
    m2.metric("Estimators", "400")
    m3.metric("Contamination", "5%")
    st.caption(f"Model input: {len(fit_df):,} complete four-parameter rows · median imputation · RobustScaler · random state 42")

with tabs[3]:
    st.subheader("Dataset Explorer")
    st.markdown("<div class='section-note'>Browse, inspect and export the complete four-parameter process dataset.</div>", unsafe_allow_html=True)
    st.write("")
    e1,e2,e3 = st.columns(3)
    e1.metric("Available records", f"{len(fit_df):,}")
    e2.metric("Displayed columns", "06")
    e3.metric("Process parameters", "04")
    display = fit_df[["training_row_id", "input_group"] + FEATURES].copy()
    display = display.rename(columns={"training_row_id":"Record ID", "input_group":"Record group", **FEATURE_LABELS})
    page_size = st.selectbox("Rows per page", [25, 50, 100, 250, 500], index=2, key="explorer_page_size")
    total_pages = max(1, int(np.ceil(len(display) / page_size)))
    page_options = list(range(1, total_pages + 1))
    page = st.selectbox("Page", page_options, index=0, key="explorer_page_select")
    start = (int(page) - 1) * page_size
    page_df = display.iloc[start:start + page_size]
    st.dataframe(page_df, use_container_width=True, height=470, hide_index=True)
    st.caption(f"Showing {start + 1 if len(display) else 0}–{min(start + page_size, len(display))} of {len(display):,} matching records · Page {page} of {total_pages}")
    st.download_button("Export complete parameter dataset (CSV)", fit_df.to_csv(index=False).encode("utf-8"),
                       "complete_process_parameter_records.csv", "text/csv", use_container_width=True)
    st.subheader("Parameter relationships")
    if len(fit_df):
        fig = px.scatter(fit_df, x=FEATURES[0], y=FEATURES[1], color=FEATURES[2], symbol="input_group",
                         hover_data=["training_row_id", "input_group"], labels=FEATURE_LABELS,
                         title="Pressure–speed relationship across all parameter records", template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("The parameter dataset is currently unavailable.")

with tabs[4]:
    st.subheader("Parameter Lab")
    st.markdown("<div class='section-note'>Investigate operating windows and compare parameter combinations across the available scenario space.</div>", unsafe_allow_html=True)
    if len(scenarios_df):
        max_rows = min(5000, len(scenarios_df))
        n = st.slider("Scenario points to visualise", min_value=100, max_value=max_rows, value=min(1500, max_rows), step=100)
        sample = scenarios_df.head(n).copy()
        q1,q2,q3 = st.columns(3)
        q1.metric("Scenario entries", f"{len(scenarios_df):,}")
        q2.metric("Visualised points", f"{len(sample):,}")
        q3.metric("Parameter dimensions", "04")
        fig = px.scatter(sample, x="pressure_kpa_scenario", y="speed_mm_s_scenario", color="temperature_c_scenario",
                         hover_data=["scenario_id", "source_record_id", "components_context"],
                         labels={"pressure_kpa_scenario":"Pressure (kPa)", "speed_mm_s_scenario":"Speed (mm/s)", "temperature_c_scenario":"Temperature (°C)"},
                         title="Pressure–speed operating space", template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sample.head(200), use_container_width=True, height=380, hide_index=True)
        st.download_button("Export scenario table (CSV)", scenarios_df.to_csv(index=False).encode("utf-8"),
                           "parameter_space_scenarios.csv", "text/csv", use_container_width=True)
    else:
        st.error("Scenario table is unavailable. Check the processed data folder.")

with tabs[5]:
    st.subheader("Methodology & Sources")
    st.markdown("**Data processing pipeline**")
    st.markdown("1. Ingest the CECT open-source bioink database and retain source-linked metadata.\n2. Standardise process fields into pressure (kPa), printing speed (mm/s), temperature (°C) and nozzle diameter (µm).\n3. Assemble the complete four-parameter model-fit table for multivariate analysis.\n4. Apply median imputation and robust scaling before Isolation Forest screening.\n5. Explore the parameter space and screen user-defined operating points.")
    st.markdown("**Model interpretation**")
    st.write("The current model is an unsupervised Isolation Forest. It identifies operating points that are statistically unusual relative to the fitted parameter distribution. An anomaly flag is not a defect label, a print-quality score or a prediction of successful skin-tissue fabrication.")
    st.markdown("**Data interpretation**")
    st.write("The model-fit table combines complete source-linked parameter records with parameter-space samples derived from source-linked process settings. These two record bases remain distinguishable in the explorer and exports. The current CECT data do not provide measured filament-width or print-quality target labels, so supervised quality prediction is not claimed.")
    st.markdown("**Source references**")
    st.markdown("- [CECT Open-Source Bioink Database](https://cect.umd.edu/database) · [Associated publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC9652762/)\n- [Tansell et al. — nozzle geometry and resolution dataset (Dryad)](https://datadryad.org/dataset/doi:10.5061/dryad.866t1g21x)\n- [Bonatti et al. — extrusion bioprinting quality-control dataset (Zenodo)](https://zenodo.org/records/7024007)")
    st.info("External datasets are listed as research references; they are not represented as integrated model inputs in this build.")

st.markdown("---")
st.markdown("<div class='section-note'>FYP2026-27-013 · Skin Bioprinting Digital Twin · Process intelligence prototype</div>", unsafe_allow_html=True)
