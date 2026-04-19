"""
assessment_engine.py
--------------------
Core inference and analytics engine.
Accepts raw vineyard chemical properties, runs the XGBoost model,
and computes the full assessment suite: quality score, percentile,
maturity, risk, stability, recommendations, and feature importance.
"""

import logging
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Thresholds derived from domain knowledge and dataset statistics
QUALITY_SCALE_MIN = 3.0
QUALITY_SCALE_MAX = 9.0

# Optimal ranges per feature (based on domain enology knowledge)
OPTIMAL_RANGES = {
    "fixed acidity":        (6.0, 9.0),
    "volatile acidity":     (0.2, 0.5),
    "citric acid":          (0.25, 0.5),
    "residual sugar":       (1.0, 5.0),
    "chlorides":            (0.03, 0.08),
    "free sulfur dioxide":  (15.0, 40.0),
    "total sulfur dioxide": (30.0, 120.0),
    "density":              (0.990, 0.999),
    "pH":                   (3.1, 3.5),
    "sulphates":            (0.4, 0.8),
    "alcohol":              (10.0, 13.0),
}

FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def _normalize_quality_score(predicted: float) -> float:
    """
    Map predicted quality (3–9 scale) to a 0–100 percentage score.
    """
    clamped = max(QUALITY_SCALE_MIN, min(QUALITY_SCALE_MAX, predicted))
    score = (clamped - QUALITY_SCALE_MIN) / (QUALITY_SCALE_MAX - QUALITY_SCALE_MIN) * 100
    return round(score, 2)


def _compute_percentile(predicted_quality: float, quality_array: np.ndarray) -> float:
    """
    Compute what percentile the predicted quality sits at
    within the training dataset distribution.
    """
    percentile = stats.percentileofscore(quality_array, predicted_quality, kind="rank")
    return round(percentile, 2)


def _assess_maturity(features: dict) -> tuple:
    """
    Determine vineyard maturity based on chemical indicators:
    - Alcohol (primary driver of ripeness)
    - pH (rises as grapes ripen)
    - Fixed acidity (drops as grapes ripen)
    Returns (maturity_index: float, maturity_status: str).
    """
    alcohol = features.get("alcohol", 10.0)
    ph = features.get("pH", 3.2)
    fixed_acidity = features.get("fixed acidity", 7.0)

    # Normalized ripeness sub-scores (0–1 each)
    alcohol_score = np.clip((alcohol - 8.0) / (14.0 - 8.0), 0, 1)
    ph_score = np.clip((ph - 2.8) / (3.8 - 2.8), 0, 1)
    acidity_score = np.clip(1.0 - (fixed_acidity - 4.0) / (14.0 - 4.0), 0, 1)

    maturity_index = round((alcohol_score * 0.5 + ph_score * 0.25 + acidity_score * 0.25) * 100, 2)

    if maturity_index < 35:
        status = "Under-Mature"
    elif maturity_index < 65:
        status = "Developing"
    elif maturity_index < 85:
        status = "Peak Maturity"
    else:
        status = "Over-Mature"

    return maturity_index, status


def _compute_risk(features: dict, feature_importance: dict) -> tuple:
    """
    Compute risk score based on how far each feature deviates
    from its optimal range, weighted by feature importance.
    Returns (risk_percentage: float, risk_severity: str).
    """
    total_weight = 0.0
    weighted_deviation = 0.0

    for feature, (low, high) in OPTIMAL_RANGES.items():
        value = features.get(feature, (low + high) / 2)
        importance = feature_importance.get(feature, 1.0 / len(FEATURE_NAMES))

        if value < low:
            deviation = (low - value) / max(abs(low), 1e-6)
        elif value > high:
            deviation = (value - high) / max(abs(high), 1e-6)
        else:
            deviation = 0.0

        deviation = min(deviation, 2.0)  # Cap at 200% deviation
        weighted_deviation += deviation * importance
        total_weight += importance

    if total_weight > 0:
        raw_risk = weighted_deviation / total_weight
    else:
        raw_risk = 0.0

    risk_percentage = round(min(raw_risk * 100, 100.0), 2)

    if risk_percentage < 30:
        severity = "Low"
    elif risk_percentage < 60:
        severity = "Medium"
    else:
        severity = "High"

    return risk_percentage, severity


def _assess_stability(features: dict, risk_percentage: float) -> str:
    """
    Determine chemical stability based on sulfur dioxide balance
    and overall risk level.
    Returns 'Stable' or 'Unstable'.
    """
    free_so2 = features.get("free sulfur dioxide", 30)
    total_so2 = features.get("total sulfur dioxide", 80)
    volatile_acidity = features.get("volatile acidity", 0.4)

    # Sulfur dioxide ratio (below ~0.7 suggests poor preservation balance)
    so2_ratio = free_so2 / max(total_so2, 1.0)
    so2_ok = 0.15 <= so2_ratio <= 0.75

    va_ok = volatile_acidity <= 0.7

    if so2_ok and va_ok and risk_percentage < 65:
        return "Stable"
    return "Unstable"


def _generate_recommendations(features: dict, feature_importance: dict) -> list:
    """
    Generate actionable recommendations by identifying the top deviating
    features weighted by their model importance.
    Returns a list of recommendation strings (up to 5).
    """
    deviations = []

    for feature, (low, high) in OPTIMAL_RANGES.items():
        value = features.get(feature)
        if value is None:
            continue

        importance = feature_importance.get(feature, 0.0)

        if value < low:
            direction = "Increase"
            magnitude = low - value
        elif value > high:
            direction = "Reduce"
            magnitude = value - high
        else:
            continue

        deviations.append((importance * magnitude, direction, feature, value, low, high))

    # Sort by weighted severity, descending
    deviations.sort(reverse=True)

    recommendations = []
    for _, direction, feature, value, low, high in deviations[:5]:
        if direction == "Increase":
            rec = f"{direction} {feature} from {value:.2f} toward optimal range [{low:.2f} – {high:.2f}]"
        else:
            rec = f"{direction} {feature} from {value:.2f} toward optimal range [{low:.2f} – {high:.2f}]"
        recommendations.append(rec)

    if not recommendations:
        recommendations.append("Chemical profile is within optimal ranges — no major adjustments required.")

    return recommendations


def run_assessment(
    features: dict,
    model,
    quality_array: list,
    feature_importance: dict,
) -> dict:
    """
    Full assessment pipeline.

    Args:
        features         : dict of feature_name → float value
        model            : trained XGBRegressor
        quality_array    : array of quality scores from training data
        feature_importance: dict of feature_name → importance score

    Returns:
        Full assessment result dict matching the output specification.
    """
    # Build input vector in correct feature order
    input_vector = np.array([[features[f] for f in FEATURE_NAMES]], dtype=np.float64)

    # Predict directly without scaling
    predicted_quality = float(model.predict(input_vector)[0])
    predicted_quality = round(max(QUALITY_SCALE_MIN, min(QUALITY_SCALE_MAX, predicted_quality)), 4)

    quality_score_pct = _normalize_quality_score(predicted_quality)
    quality_percentile = _compute_percentile(predicted_quality, quality_array)
    maturity_index, maturity_status = _assess_maturity(features)
    risk_percentage, risk_severity = _compute_risk(features, feature_importance)
    stability = _assess_stability(features, risk_percentage)
    recommendations = _generate_recommendations(features, feature_importance)

    result = {
        "predicted_quality": round(predicted_quality, 4),
        "quality_score_pct": quality_score_pct,
        "quality_percentile": quality_percentile,
        "maturity_index": maturity_index,
        "maturity_status": maturity_status,
        "risk_percentage": risk_percentage,
        "risk_severity": risk_severity,
        "stability": stability,
        "recommendations": recommendations,
        "feature_importance": feature_importance,
    }

    logger.info(
        "Assessment complete — Quality: %.2f | Percentile: %.1f%% | Risk: %s | Stability: %s",
        predicted_quality,
        quality_percentile,
        risk_severity,
        stability,
    )

    return result
