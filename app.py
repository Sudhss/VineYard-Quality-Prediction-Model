import streamlit as st
import pandas as pd
import joblib

from src.assessment_engine import full_wine_assessment

# ------------------------------------------------------------
# App Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Vineyard Quality Assessment",
    layout="wide"
)

# ------------------------------------------------------------
# Load model & artifacts (cached)
# ------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("model/random_forest_model.joblib")
    artifacts = joblib.load("model/artifacts.joblib")
    return model, artifacts

model, artifacts = load_assets()

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# ------------------------------------------------------------
# Quality interpretation helper
# ------------------------------------------------------------
def interpret_quality(score_percent):
    if score_percent >= 90:
        return "Very Good", "#2ecc71", "Excellent yield expected"
    elif score_percent >= 70:
        return "Moderately Good", "#27ae60", "Good yield with minor optimizations"
    elif score_percent >= 50:
        return "Average", "#f1c40f", "Average yield, improvements recommended"
    elif score_percent >= 30:
        return "Slightly Bad", "#e67e22", "Low yield, intervention needed"
    else:
        return "Very Bad", "#e74c3c", "High risk, poor yield expected"

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("🍇 Vineyard Quality Assessment System")
st.caption("ML-powered vineyard quality prediction and decision intelligence")

# ============================================================
# INPUT VIEW
# ============================================================
if not st.session_state.show_results:

    st.subheader("Enter Vineyard Chemical Properties")

    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 7.4)
        volatile_acidity = st.number_input("Volatile Acidity", 0.1, 2.0, 0.7)
        citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
        residual_sugar = st.number_input("Residual Sugar", 0.5, 15.0, 1.9)

    with col2:
        chlorides = st.number_input("Chlorides", 0.01, 0.6, 0.076)
        free_so2 = st.number_input("Free Sulfur Dioxide", 1.0, 75.0, 11.0)
        total_so2 = st.number_input("Total Sulfur Dioxide", 6.0, 300.0, 34.0)
        density = st.number_input("Density", 0.990, 1.005, 0.9978)

    with col3:
        pH = st.number_input("pH", 2.7, 4.1, 3.51)
        sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.56)
        alcohol = st.number_input("Alcohol (%)", 8.0, 15.0, 9.4)

    if st.button("Analyze Vineyard"):
        st.session_state.user_input = {
            "fixed acidity": fixed_acidity,
            "volatile acidity": volatile_acidity,
            "citric acid": citric_acid,
            "residual sugar": residual_sugar,
            "chlorides": chlorides,
            "free sulfur dioxide": free_so2,
            "total sulfur dioxide": total_so2,
            "density": density,
            "pH": pH,
            "sulphates": sulphates,
            "alcohol": alcohol
        }
        st.session_state.show_results = True
        st.rerun()

# ============================================================
# RESULTS VIEW
# ============================================================
else:
    input_features = pd.Series(st.session_state.user_input)

    result = full_wine_assessment(
        model=model,
        input_features=input_features,
        high_quality_stats=artifacts["high_quality_stats"],
        low_quality_benchmarks=artifacts["low_quality_benchmarks"],
        historical_quality_scores=artifacts["historical_quality_scores"],
        mean_y_train=artifacts["mean_y_train"],
        std_dev_residuals=artifacts["std_dev_residuals"],
        perturbation_percentage=0.05,
        sorted_importance_df=artifacts["sorted_importance_df"]
    )

    metrics = result["multi_dimensional_metrics"]

    # --------------------------------------------------------
    # OVERALL QUALITY BLOCK
    # --------------------------------------------------------
    quality_percent = (metrics["predicted_quality"] / 8.0) * 100
    label, color, yield_msg = interpret_quality(quality_percent)

    st.subheader("Overall Vineyard Quality")

    with st.container():
        col1, col2 = st.columns([1, 3])

        with col1:
            st.metric(
                label="Quality Score",
                value=f"{quality_percent:.1f}%",
                delta=label
            )

        with col2:
            st.progress(min(quality_percent / 100, 1.0))
            st.write(f"**Yield Insight:** {yield_msg}")




    st.progress(min(quality_percent / 100, 1.0))
    st.divider()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    st.subheader("Assessment Metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Quality", f"{metrics['predicted_quality']:.2f}")
    m2.metric("Quality Percentile", f"{metrics['quality_percentile']:.1f}%")
    m3.metric("Risk Severity", metrics["risk_severity"])
    m4.metric("Stability", metrics["stability_interpretation"])

    st.divider()

    # --------------------------------------------------------
    # VISUALS
    # --------------------------------------------------------
    st.subheader("Risk & Stability")

    st.write("Risk Index")
    st.progress(metrics["risk_percentage"] / 100)

    st.write("Stability Score")
    st.progress(metrics["stability_score"] / 100)

    st.divider()

    st.subheader("Key Drivers of Quality")
    st.bar_chart(artifacts["sorted_importance_df"].head(5))

    st.divider()

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------
    st.subheader("Chemical Property Recommendations")
    for feature, advice in result["chemical_property_recommendations"].items():
        st.write(f"- **{feature}**: {advice}")

    st.divider()

    st.subheader("Final Decision")
    st.success(result["overall_decision"])

    st.divider()

    st.subheader("Raw JSON Output")
    st.json(result)

    if st.button("Analyze Another Vineyard"):
        st.session_state.show_results = False
        st.rerun()
