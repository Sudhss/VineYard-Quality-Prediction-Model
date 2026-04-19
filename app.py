import sys
import os
import json
import logging
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model_engine import load_model_and_artifacts, get_feature_importance
from assessment_engine import run_assessment, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vineyard Quality Assessment",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES — Apple light theme + glassmorphism
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background: #f5f2ee !important;
        color: #1a1714 !important;
    }
    /* Only override font on streamlit elements, NOT background */
    [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f5f2ee 0%, #ede8e1 40%, #e8e2d8 100%) !important;
        min-height: 100vh;
    }

    /* Remove default streamlit padding */
    .block-container {
        padding: 2rem 3rem 4rem 3rem !important;
        max-width: 1280px;
    }

    /* ── Header ── */
    .vqa-header {
        display: flex;
        align-items: flex-end;
        gap: 1.5rem;
        margin-bottom: 2.5rem;
        padding-bottom: 2rem;
        border-bottom: 1px solid rgba(26,23,20,0.08);
    }

    .vqa-brand-title {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2.6rem !important;
        font-weight: 400 !important;
        letter-spacing: -0.02em !important;
        color: #1a1714 !important;
        -webkit-text-fill-color: #1a1714 !important;
        line-height: 1 !important;
        margin: 0 !important;
    }

    .vqa-brand-subtitle {
        font-size: 0.9rem;
        font-weight: 400;
        color: #7a7168;
        margin: 0 0 0.2rem 0;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .vqa-version-badge {
        background: rgba(26,23,20,0.07);
        border-radius: 100px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 500;
        color: #5a5349;
        margin-left: auto;
        align-self: center;
    }

    /* ── Glass cards ── */
    .glass-card {
        background: #ffffff !important;
        border-radius: 20px !important;
        border: 1px solid rgba(26,23,20,0.10) !important;
        box-shadow:
            0 8px 32px rgba(26,23,20,0.10),
            0 2px 8px rgba(26,23,20,0.06) !important;
        padding: 2rem !important;
        margin-bottom: 1.25rem !important;
    }

    .glass-card-sm {
        background: #ffffff !important;
        border-radius: 16px !important;
        border: 1px solid rgba(26,23,20,0.10) !important;
        box-shadow:
            0 8px 32px rgba(26,23,20,0.10),
            0 2px 8px rgba(26,23,20,0.06) !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9a9088;
        margin-bottom: 0.4rem;
    }

    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        font-weight: 400;
        color: #1a1714;
        margin: 0 0 1.5rem 0;
        line-height: 1.2;
    }

    /* ── Quality score display ── */
    .quality-score-number {
        font-family: 'DM Serif Display', serif;
        font-size: 4.5rem;
        font-weight: 400;
        color: #1a1714;
        line-height: 1;
        margin: 0.5rem 0;
        letter-spacing: -0.02em;
    }

    .quality-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .badge-excellent { background: #d4edd8; color: #2d6b38; }
    .badge-good      { background: #d8ead4; color: #3a6b2d; }
    .badge-average   { background: #e8e4d0; color: #6b5e2d; }
    .badge-below     { background: #ead4d4; color: #6b2d2d; }

    /* ── Progress bar ── */
    .progress-track {
        background: rgba(26,23,20,0.08);
        border-radius: 100px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .progress-fill {
        height: 100%;
        border-radius: 100px;
        transition: width 0.6s ease;
    }

    .fill-excellent { background: linear-gradient(90deg, #4ade80, #22c55e); }
    .fill-good      { background: linear-gradient(90deg, #86efac, #4ade80); }
    .fill-average   { background: linear-gradient(90deg, #c4a55a, #a08030); }
    .fill-below     { background: linear-gradient(90deg, #f87171, #dc2626); }

    /* ── Metric cards ── */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.55);
        border: 1px solid rgba(255,255,255,0.8);
        border-radius: 16px;
        padding: 1rem 0.75rem; /* Reduced padding to give text more space */
        box-shadow: 0 2px 12px rgba(26,23,20,0.04);
        text-align: center;
    }

    .metric-label {
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #9a9088;
        margin-bottom: 0.4rem;
        white-space: nowrap;
    }

    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 1.75rem;
        font-weight: 400;
        color: #1a1714;
        line-height: 1.1;
        letter-spacing: -0.01em;
        white-space: nowrap;
    }

    .metric-sub {
        font-size: 0.74rem;
        color: #7a7168;
        margin-top: 0.25rem;
        white-space: nowrap;
    }

    /* ── Risk badge colours ── */
    .risk-low    { color: #2d6b38; }
    .risk-medium { color: #6b5e2d; }
    .risk-high   { color: #8b2020; }

    /* ── Stability badge ── */
    .stability-stable   { color: #2d526b; }
    .stability-unstable { color: #6b2d2d; }

    /* ── Input panel ── */
    .input-section-title {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.8rem !important;
        font-weight: 400 !important;
        color: #1a1714 !important;
        -webkit-text-fill-color: #1a1714 !important;
        letter-spacing: -0.02em !important;
        margin: 0 0 0.4rem 0 !important;
        line-height: 1.15 !important;
    }

    .input-section-sub {
        font-size: 0.85rem;
        color: #7a7168;
        margin-bottom: 1.75rem;
    }

    /* Override Streamlit number inputs */
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.8) !important;
        border: 1px solid rgba(26,23,20,0.12) !important;
        border-radius: 10px !important;
        padding: 0.5rem 0.75rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        color: #1a1714 !important;
        box-shadow: 0 1px 4px rgba(26,23,20,0.04) !important;
    }

    .stNumberInput > div > div > input:focus {
        border-color: rgba(160, 128, 48, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(160,128,48,0.12) !important;
        outline: none !important;
    }

    /* Label styling */
    .stNumberInput label {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #5a5349 !important;
        letter-spacing: 0.02em !important;
        text-transform: none !important;
    }

    /* File uploader — light theme override */
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploadDropzone"] {
        background: #f7f5f2 !important;
        border: 1.5px dashed rgba(26,23,20,0.20) !important;
        border-radius: 14px !important;
        color: #1a1714 !important;
    }
    [data-testid="stFileUploadDropzone"] * {
        color: #1a1714 !important;
        -webkit-text-fill-color: #1a1714 !important;
    }
    /* Browse files button */
    [data-testid="stFileUploadDropzone"] button,
    [data-testid="baseButton-secondary"] {
        background: #1a1714 !important;
        color: #f5f2ee !important;
        border: none !important;
        border-radius: 8px !important;
    }

    /* ── CTA Button ── */
    .stButton > button {
        width: 100%;
        background: #1a1714 !important;
        color: #f5f2ee !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
        cursor: pointer !important;
        box-shadow: 0 4px 16px rgba(26,23,20,0.2) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #2d2a26 !important;
        box-shadow: 0 6px 20px rgba(26,23,20,0.28) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Animated Download Button ── */
    @keyframes shimmer {
        0%   { background-position: -400px 0; }
        100% { background-position: 400px 0; }
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 4px 16px rgba(160,128,48,0.25); }
        50%       { box-shadow: 0 6px 28px rgba(160,128,48,0.50); }
    }
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(
            110deg,
            #1a1714 40%,
            #3a3328 50%,
            #1a1714 60%
        ) !important;
        background-size: 800px 100% !important;
        color: #f5f2ee !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 2.5rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        cursor: pointer !important;
        animation: shimmer 2.8s infinite linear, pulse-glow 2.5s ease-in-out infinite !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        display: block !important;
        margin: 0 auto !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        animation-play-state: paused !important;
        box-shadow: 0 8px 28px rgba(26,23,20,0.35) !important;
    }
    [data-testid="stDownloadButton"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 1.5rem auto 0 auto !important;
        width: fit-content !important;
        min-width: 280px !important;
    }
    [data-testid="stDownloadButton"] > button {
        min-width: 280px !important;
    }

    /* ── Recommendation pills ── */
    .rec-pill {
        background: rgba(160,128,48,0.08);
        border: 1px solid rgba(160,128,48,0.2);
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-size: 0.85rem;
        color: #4a3e1c;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        line-height: 1.4;
    }

    .rec-bullet {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #a08030;
        margin-top: 0.4rem;
        flex-shrink: 0;
    }

    /* ── Feature importance bar ── */
    .fi-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.65rem;
    }

    .fi-label {
        font-size: 0.8rem;
        color: #5a5349;
        width: 160px;
        flex-shrink: 0;
        text-align: right;
    }

    .fi-track {
        flex: 1;
        background: rgba(26,23,20,0.07);
        border-radius: 100px;
        height: 6px;
        overflow: hidden;
    }

    .fi-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #c4a55a, #a08030);
    }

    .fi-pct {
        font-size: 0.78rem;
        color: #9a9088;
        width: 38px;
        text-align: right;
    }

    /* ── Yield insight ── */
    .yield-insight {
        font-size: 0.88rem;
        color: #5a5349;
        background: rgba(160,128,48,0.06);
        border-left: 3px solid rgba(160,128,48,0.4);
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin-top: 0.5rem;
        font-style: italic;
    }

    /* ── Maturity indicator ── */
    .maturity-bar-wrapper {
        position: relative;
        margin: 0.75rem 0;
    }

    .maturity-track {
        background: linear-gradient(90deg,
            rgba(59,130,246,0.3) 0%,
            rgba(34,197,94,0.4) 35%,
            rgba(74,222,128,0.5) 65%,
            rgba(239,68,68,0.35) 100%
        );
        border-radius: 100px;
        height: 10px;
        position: relative;
        overflow: visible;
    }

    .maturity-thumb {
        position: absolute;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 16px;
        height: 16px;
        background: white;
        border: 2.5px solid #a08030;
        border-radius: 50%;
        box-shadow: 0 2px 6px rgba(26,23,20,0.2);
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid rgba(26,23,20,0.07) !important;
        margin: 1.75rem 0 !important;
    }

    /* ── Streamlit element overrides ── */
    [data-testid="stHorizontalBlock"] { gap: 1.25rem !important; }
    [data-testid="column"] > div { gap: 0 !important; }

    /* Transparent column/vertical containers */
    [data-testid="stColumn"],
    [data-testid="stColumn"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }

    /* Hide orphan empty div tags injected by st.markdown('<div>') and st.markdown('</div>') */
    [data-testid="stMarkdown"] div.glass-card:empty,
    [data-testid="stMarkdown"] div.glass-card-sm:empty,
    [data-testid="stMarkdown"]:has(> div > .glass-card:empty),
    [data-testid="stMarkdown"]:has(> div > .glass-card-sm:empty) {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }

    /* Also hide stMarkdown wrappers that contain only a closing tag (renders as empty div) */
    [data-testid="stMarkdown"] > div:empty {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── CSV preview ── */
    .csv-preview-label {
        font-size: 0.75rem;
        color: #9a9088;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* Alert overrides */
    .stAlert {
        background: rgba(255,255,255,0.7) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(26,23,20,0.1) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model, artifacts = load_model_and_artifacts()
        return model, artifacts
    except FileNotFoundError:
        return None, None


model, artifacts = load_model()

# ─────────────────────────────────────────────
# HELPER: derive grade label
# ─────────────────────────────────────────────
def _grade(score_pct: float) -> tuple:
    if score_pct >= 75:
        return "Excellent", "badge-excellent", "fill-excellent"
    elif score_pct >= 58:
        return "Good", "badge-good", "fill-good"
    elif score_pct >= 42:
        return "Average", "badge-average", "fill-average"
    else:
        return "Below Average", "badge-below", "fill-below"


def _yield_insight(score_pct: float) -> str:
    if score_pct >= 75:
        return "Exceptional yield potential — chemical profile is optimal for premium production."
    elif score_pct >= 58:
        return "Good yield expected — minor adjustments could elevate to premium tier."
    elif score_pct >= 42:
        return "Average yield — improvements recommended to reach consistent quality."
    else:
        return "Below-average yield detected — significant chemical imbalances present."


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="vqa-header">
        <div>
            <p class="vqa-brand-subtitle">XGBoost ML System</p>
            <h1 class="vqa-brand-title">Vineyard Quality Assessment</h1>
        </div>
        <span class="vqa-version-badge">v1.0  Production</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# MODEL STATUS BANNER
# ─────────────────────────────────────────────
if model is None:
    st.error(
        "Model not found. Please run `python main.py` to train and save the model before launching the app."
    )
    st.stop()

# ─────────────────────────────────────────────
# LAYOUT: left panel (inputs) | right panel (results)
# ─────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1], gap="large")

# ═══════════════════════════════════════════
# LEFT PANEL — Input
# ═══════════════════════════════════════════
with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <p class="section-label">Data Input</p>
        <h2 class="input-section-title">Upload or Enter Vineyard Data</h2>
        """,
        unsafe_allow_html=True,
    )

    # CSV upload
    uploaded_file = st.file_uploader(
        "Or upload a CSV file",
        type=["csv"],
        help="Upload a CSV with the same columns as the UCI Wine Quality dataset.",
    )

    csv_features = None
    if uploaded_file is not None:
        try:
            csv_df = pd.read_csv(uploaded_file)
            required_cols = [f.lower().replace(" ", " ") for f in FEATURE_NAMES]

            # Normalize column names for matching
            csv_df.columns = [c.strip().lower() for c in csv_df.columns]

            # Check all features present
            missing = [f for f in FEATURE_NAMES if f not in csv_df.columns]
            if missing:
                st.warning(f"CSV missing columns: {', '.join(missing)}")
            else:
                row = csv_df.iloc[0]
                csv_features = {f: float(row[f]) for f in FEATURE_NAMES}
                st.markdown('<p class="csv-preview-label">Using first row from uploaded file</p>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        """
        <p class="section-label">Manual Entry</p>
        <p class="input-section-sub">Enter chemical properties measured from your vineyard sample.</p>
        """,
        unsafe_allow_html=True,
    )

    # Default values (representative of a premium wine)
    defaults = {
        "fixed acidity": 7.5,
        "volatile acidity": 0.25,
        "citric acid": 0.40,
        "residual sugar": 2.0,
        "chlorides": 0.040,
        "free sulfur dioxide": 30.0,
        "total sulfur dioxide": 90.0,
        "density": 0.9930,
        "pH": 3.30,
        "sulphates": 0.70,
        "alcohol": 12.5,
    }

    # Override defaults with CSV values if uploaded
    if csv_features:
        defaults.update(csv_features)

    col1, col2, col3 = st.columns(3)

    with col1:
        fa = st.number_input("Fixed Acidity", value=defaults["fixed acidity"], step=0.1, format="%.2f")
        va = st.number_input("Volatile Acidity", value=defaults["volatile acidity"], step=0.01, format="%.3f")
        ca = st.number_input("Citric Acid", value=defaults["citric acid"], step=0.01, format="%.3f")
        rs = st.number_input("Residual Sugar", value=defaults["residual sugar"], step=0.1, format="%.2f")

    with col2:
        cl = st.number_input("Chlorides", value=defaults["chlorides"], step=0.001, format="%.4f")
        fsd = st.number_input("Free Sulfur Dioxide", value=defaults["free sulfur dioxide"], step=1.0, format="%.1f")
        tsd = st.number_input("Total Sulfur Dioxide", value=defaults["total sulfur dioxide"], step=1.0, format="%.1f")
        den = st.number_input("Density", value=defaults["density"], step=0.0001, format="%.4f")

    with col3:
        ph = st.number_input("pH", value=defaults["pH"], step=0.01, format="%.2f")
        sul = st.number_input("Sulphates", value=defaults["sulphates"], step=0.01, format="%.3f")
        alc = st.number_input("Alcohol (%)", value=defaults["alcohol"], step=0.1, format="%.1f")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("Analyze Vineyard", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Training metrics card
    if artifacts and "metrics" in artifacts:
        m = artifacts["metrics"]
        st.markdown('<div class="glass-card-sm">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Model Performance</p>', unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f'<div class="metric-label">MAE</div><div style="font-size:1.4rem;font-family:\'DM Serif Display\',serif;color:#1a1714">{m["MAE"]:.2f}</div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="metric-label">RMSE</div><div style="font-size:1.4rem;font-family:\'DM Serif Display\',serif;color:#1a1714">{m["RMSE"]:.2f}</div>', unsafe_allow_html=True)
        with mc3:
            st.markdown(f'<div class="metric-label">R²</div><div style="font-size:1.4rem;font-family:\'DM Serif Display\',serif;color:#1a1714">{m["R2"]:.2f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# RIGHT PANEL — Results
# ═══════════════════════════════════════════
with right_col:

    # ── Initial placeholder state ──
    if not analyze_clicked:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <p class="section-label">Assessment Output</p>
            <h2 style="font-family:'DM Serif Display',serif;font-size:1.8rem;font-weight:400;letter-spacing:-0.02em;color:#1a1714;-webkit-text-fill-color:#1a1714;margin:0 0 0.75rem 0;line-height:1.15">Overall Vineyard Quality</h2>
            <p style="color:#9a9088;font-size:0.9rem;margin-top:2rem;">
                Enter chemical properties and click <strong>Analyze Vineyard</strong>
                to generate a complete quality assessment.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # ── Build features dict ──
        features = {
            "fixed acidity": fa,
            "volatile acidity": va,
            "citric acid": ca,
            "residual sugar": rs,
            "chlorides": cl,
            "free sulfur dioxide": fsd,
            "total sulfur dioxide": tsd,
            "density": den,
            "pH": ph,
            "sulphates": sul,
            "alcohol": alc,
        }

        # ── Run assessment ──
        with st.spinner("Running XGBoost inference..."):
            try:
                fi = get_feature_importance(model)
                
                # Fallback in case artifacts.joblib is outdated
                q_arr = artifacts.get("quality_distribution", [5, 6, 7])
                
                result = run_assessment(
                    features=features,
                    model=model,
                    quality_array=q_arr,
                    feature_importance=fi,
                )
            except Exception as exc:
                st.error(f"Assessment failed: {exc}")
                logger.exception("Assessment error")
                st.stop()

        # ── Save output JSON ──
        os.makedirs("output", exist_ok=True)
        output_data = {
            "predicted_quality": result["predicted_quality"],
            "quality_percentile": result["quality_percentile"],
            "maturity_index": result["maturity_index"],
            "maturity_status": result["maturity_status"],
            "risk_percentage": result["risk_percentage"],
            "risk_severity": result["risk_severity"],
            "stability": result["stability"],
            "recommendations": result["recommendations"],
            "feature_importance": result["feature_importance"],
        }
        with open("output/assessment_result.json", "w") as f:
            json.dump(output_data, f, indent=2)

        score_pct = result["quality_score_pct"]
        grade, badge_class, fill_class = _grade(score_pct)
        insight = _yield_insight(score_pct)

        # ─── DASHBOARD CARD 1: Overall Quality ───
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown('<p class="section-label">Assessment Output</p><h2 style="font-family:\'DM Serif Display\',serif;font-size:1.8rem;font-weight:400;letter-spacing:-0.02em;color:#1a1714;-webkit-text-fill-color:#1a1714;margin:0 0 0.75rem 0;line-height:1.15">Overall Vineyard Quality</h2>', unsafe_allow_html=True)

        d1, d2 = st.columns([1, 1.6])
        with d1:
            st.markdown(f'<p style="font-size:0.78rem;color:#9a9088;margin:0 0 0.2rem 0">Quality Score</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="quality-score-number">{score_pct:.1f}%</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="quality-badge {badge_class}">{grade}</span>', unsafe_allow_html=True)

        with d2:
            st.markdown(
                f"""
                <div style="margin-top:0.5rem">
                    <div class="progress-track">
                        <div class="progress-fill {fill_class}" style="width:{score_pct}%"></div>
                    </div>
                    <div class="yield-insight">
                        <strong>Yield Insight:</strong> {insight}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ─── Metric cards ───
        st.markdown('<p class="section-label">Assessment Metrics</p>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Quality</div>
                    <div class="metric-value">{result['predicted_quality']:.2f}</div>
                    <div class="metric-sub">out of 10 scale</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Quality Percentile</div>
                    <div class="metric-value">{result['quality_percentile']:.1f}%</div>
                    <div class="metric-sub">vs. dataset</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        risk_class = f"risk-{result['risk_severity'].lower()}"
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Severity</div>
                    <div class="metric-value {risk_class}">{result['risk_severity']}</div>
                    <div class="metric-sub">{result['risk_percentage']:.1f}% risk score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        stab_class = f"stability-{result['stability'].lower()}"
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Stability</div>
                    <div class="metric-value {stab_class}">{result['stability']}</div>
                    <div class="metric-sub">chemical balance</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ─── DASHBOARD CARD 2: Maturity + Recommendations ───
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)

        with mc1:
            st.markdown(
                f"""
                <p class="section-label">Maturity Analysis</p>
                <p style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:#1a1714;margin:0 0 0.5rem 0">{result['maturity_status']}</p>
                <p style="font-size:0.85rem;color:#7a7168;margin:0 0 0.75rem 0">Maturity Index: <strong>{result['maturity_index']:.1f}</strong> / 100</p>
                <div class="maturity-track">
                    <div class="maturity-thumb" style="left:{result['maturity_index']}%"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#9a9088;margin-top:0.4rem">
                    <span>Under-Mature</span><span>Peak</span><span>Over-Mature</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with mc2:
            st.markdown('<p class="section-label">Recommendations</p>', unsafe_allow_html=True)
            for rec in result["recommendations"]:
                st.markdown(
                    f'<div class="rec-pill"><div class="rec-bullet"></div>{rec}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)


# ─── FULL-WIDTH: Feature Importance + Download (outside columns) ───
if 'result' in dir() and analyze_clicked:
    st.markdown('<div class="glass-card" style="margin-top:1.25rem">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Explainability</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'DM Serif Display\',serif;font-size:1.4rem;font-weight:400;color:#1a1714;-webkit-text-fill-color:#1a1714;margin:0 0 1.25rem 0;letter-spacing:-0.01em">Feature Importance</p>', unsafe_allow_html=True)

    fi_sorted = result["feature_importance"]
    max_fi = max(fi_sorted.values()) if fi_sorted else 1.0
    fi_col1, fi_col2 = st.columns(2)
    items = list(fi_sorted.items())
    half = len(items) // 2 + len(items) % 2

    with fi_col1:
        for feat, score in items[:half]:
            bar_pct = round((score / max_fi) * 100, 1)
            st.markdown(
                f"""
                <div class="fi-row">
                    <div class="fi-label">{feat}</div>
                    <div class="fi-track"><div class="fi-fill" style="width:{bar_pct}%"></div></div>
                    <div class="fi-pct">{score*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

    with fi_col2:
        for feat, score in items[half:]:
            bar_pct = round((score / max_fi) * 100, 1)
            st.markdown(
                f"""
                <div class="fi-row">
                    <div class="fi-label">{feat}</div>
                    <div class="fi-track"><div class="fi-fill" style="width:{bar_pct}%"></div></div>
                    <div class="fi-pct">{score*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Center download button using Streamlit columns
    _, center_col, _ = st.columns([1, 1.5, 1])
    with center_col:
        st.download_button(
            label="⬇  Download Assessment JSON",
            data=json.dumps(output_data, indent=2),
            file_name="vineyard_assessment.json",
            mime="application/json",
            use_container_width=True
        )
