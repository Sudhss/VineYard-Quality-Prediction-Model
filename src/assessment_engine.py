import numpy as np
from scipy import stats

# ------------------------------------------------------------------------------------------------------------------------------------------------
# calculate_quality_percentile
# ------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_quality_percentile(predicted_quality, historical_quality_scores):
    historical_quality_scores = np.array(historical_quality_scores).flatten()
    if predicted_quality < historical_quality_scores.min():
        return 0.0
    elif predicted_quality > historical_quality_scores.max():
        return 100.0
    else:
        return stats.percentileofscore(historical_quality_scores, predicted_quality, kind='rank')

# ------------------------------------------------------------------------------------------------------------------------------------------------
# calculate_maturity_index
# ------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_maturity_index(input_features, high_quality_stats):
    input_alcohol = input_features['alcohol']
    input_pH = input_features['pH']
    input_residual_sugar = input_features['residual sugar']

    hq_mean_alcohol = high_quality_stats.loc['mean', 'alcohol']
    hq_std_alcohol = high_quality_stats.loc['std', 'alcohol']
    hq_mean_pH = high_quality_stats.loc['mean', 'pH']
    hq_std_pH = high_quality_stats.loc['std', 'pH']
    hq_mean_rs = high_quality_stats.loc['mean', 'residual sugar']
    hq_std_rs = high_quality_stats.loc['std', 'residual sugar']

    alcohol_score = max(0, 1 - abs(input_alcohol - hq_mean_alcohol) / (hq_std_alcohol * 2))
    pH_score = max(0, 1 - abs(input_pH - hq_mean_pH) / (hq_std_pH * 2))
    residual_sugar_score = max(0, 1 - abs(input_residual_sugar - hq_mean_rs) / (hq_std_rs * 2))

    maturity_index = ((alcohol_score + pH_score + residual_sugar_score) / 3) * 100

    maturity_status = "Developing"
    if (input_alcohol < hq_mean_alcohol - 1.5 * hq_std_alcohol) and \
       (input_residual_sugar > hq_mean_rs + 1.5 * hq_std_rs):
        maturity_status = "Under-mature"
    elif (input_alcohol > hq_mean_alcohol + 1.5 * hq_std_alcohol) and \
         (input_pH > hq_mean_pH + 1.5 * hq_std_pH):
        maturity_status = "Over-mature"
    elif maturity_index >= 70:
        maturity_status = "Peak Maturity"

    return maturity_index, maturity_status


# ------------------------------------------------------------------------------------------------------------------------------------------------
# calculate_risk_degradation_index
# ------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_risk_degradation_index(input_features, low_quality_benchmarks):
    risk_features = ['volatile acidity', 'chlorides', 'total sulfur dioxide', 'pH']
    risk_scores = []

    for feature in risk_features:
        input_value = input_features[feature]
        lq_mean = low_quality_benchmarks.loc['mean', feature]
        lq_std = low_quality_benchmarks.loc['std', feature]

        if lq_std == 0:
            risk_score = 1.0 if input_value == lq_mean else 0.0
        else:
            risk_score = max(0, 1 - abs(input_value - lq_mean) / (lq_std * 2))
        risk_scores.append(risk_score)

    risk_percentage = (sum(risk_scores) / len(risk_scores)) * 100

    if risk_percentage < 30:
        severity = "Low"
    elif 30 <= risk_percentage < 60:
        severity = "Medium"
    else:
        severity = "High"

    return risk_percentage, severity



# ------------------------------------------------------------------------------------------------------------------------------------------------
# calculate_stability_score
# ------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_stability_score(model, input_features, perturbation_percentage):
    perturbed_predictions = []

    original_prediction = model.predict(input_features.to_frame().T)[0]
    perturbed_predictions.append(original_prediction)

    for feature_name in input_features.index:
        original_value = input_features[feature_name]

        perturbed_features_increase = input_features.copy()
        perturbed_features_increase[feature_name] = original_value * (1 + perturbation_percentage)
        pred_increase = model.predict(perturbed_features_increase.to_frame().T)[0]
        perturbed_predictions.append(pred_increase)

        perturbed_features_decrease = input_features.copy()
        perturbed_features_decrease[feature_name] = original_value * (1 - perturbation_percentage)
        pred_decrease = model.predict(perturbed_features_decrease.to_frame().T)[0]
        perturbed_predictions.append(pred_decrease)

    perturbed_predictions_array = np.array(perturbed_predictions)

    std_dev_predictions = np.std(perturbed_predictions_array)

    stability_score = max(0, 100 - (std_dev_predictions / 0.5) * 100)

    if stability_score >= 70:
        stability_interpretation = "Stable"
    else:
        stability_interpretation = "Sensitive"

    return stability_score, stability_interpretation


# ------------------------------------------------------------------------------------------------------------------------------------------------
# recommend_chemical_properties
# ------------------------------------------------------------------------------------------------------------------------------------------------
def recommend_chemical_properties(
    input_features,
    predicted_quality,
    high_quality_stats,
    low_quality_benchmarks,
    sorted_importance_df
):
    recommendations = {}
    top_features = sorted_importance_df.index[:5]

    for feature in top_features:
        input_value = input_features[feature]

        hq_mean = high_quality_stats.loc['mean', feature] if feature in high_quality_stats.columns else None
        hq_std = high_quality_stats.loc['std', feature] if feature in high_quality_stats.columns else None

        lq_mean = low_quality_benchmarks.loc['mean', feature] if feature in low_quality_benchmarks.columns else None
        lq_std = low_quality_benchmarks.loc['std', feature] if feature in low_quality_benchmarks.columns else None

        if hq_mean is not None and lq_mean is not None:
            if predicted_quality < 5.5:
                if input_value > lq_mean + lq_std:
                    recommendations[feature] = f"Reduce {feature} from {input_value:.2f} to move away from low-quality characteristics (low-quality mean: {lq_mean:.2f})."
                elif input_value < lq_mean - lq_std:
                    recommendations[feature] = f"Increase {feature} from {input_value:.2f} to move away from low-quality characteristics (low-quality mean: {lq_mean:.2f})."
                else:
                    recommendations[feature] = f"Adjust {feature} from {input_value:.2f} to optimize. Current value is near low-quality mean ({lq_mean:.2f}). Consider moving towards high-quality range."
            else:
                if input_value > hq_mean + hq_std:
                    recommendations[feature] = f"Reduce {feature} from {input_value:.2f} to align with high-quality characteristics (high-quality mean: {hq_mean:.2f})."
                elif input_value < hq_mean - hq_std:
                    recommendations[feature] = f"Increase {feature} from {input_value:.2f} to align with high-quality characteristics (high-quality mean: {hq_mean:.2f})."
                else:
                    recommendations[feature] = f"Maintain current {feature} at {input_value:.2f}. It is within an optimal range for high-quality wines (high-quality mean: {hq_mean:.2f})."
        elif hq_mean is not None:
            recommendations[feature] = f"Optimal range based on high-quality wines: Mean={hq_mean:.2f}, Std={hq_std:.2f}. Current: {input_value:.2f}."
        elif lq_mean is not None:
            recommendations[feature] = f"Consider adjusting {feature}. Current: {input_value:.2f}. Low-quality range: Mean={lq_mean:.2f}, Std={lq_std:.2f}."
        else:
            recommendations[feature] = f"No specific benchmark for {feature} available for recommendation."

    return recommendations

# ------------------------------------------------------------------------------------------------------------------------------------------------
# make_overall_decision
# ------------------------------------------------------------------------------------------------------------------------------------------------

def make_overall_decision(assessment_results):
    predicted_quality = assessment_results['predicted_quality']
    maturity_status = assessment_results['maturity_status']
    risk_severity = assessment_results['risk_severity']
    stability_interpretation = assessment_results['stability_interpretation']

    decision = ""

    if predicted_quality >= 7.0:
        if maturity_status == "Peak Maturity" and risk_severity == "Low" and stability_interpretation == "Stable":
            decision = "Excellent Quality: Ready for consumption or premium aging. Optimal balance and stability."
        elif maturity_status == "Developing" and risk_severity == "Low":
            decision = "High Potential: Further aging recommended to reach peak maturity. Low risk and good stability."
        else:
            decision = "Good Quality: Consider specific handling based on maturity, risk, or stability factors. Further analysis needed."
    elif predicted_quality >= 5.5:
        if risk_severity == "High" or stability_interpretation == "Sensitive":
            decision = "Moderate Quality with Concerns: Address high risk factors or improve stability. Not ideal for long-term aging."
        elif maturity_status == "Under-mature":
            decision = "Moderate Quality, Under-mature: Needs more time for development, or consider specific treatments to enhance maturity."
        else:
            decision = "Solid Mid-Range Quality: Suitable for general consumption. No immediate concerns but lacks premium characteristics."
    else:
        if risk_severity == "High":
            decision = "Poor Quality, High Risk: Significant issues detected. Not recommended for consumption or requires extensive intervention."
        elif maturity_status == "Over-mature":
            decision = "Poor Quality, Over-mature: Past its prime, likely degraded. Consider alternative uses or discard."
        else:
            decision = "Improvement Recommended: The current quality is below optimal. Implement immediate interventions to improve chemical balance, or consider alternative uses."

    return decision


# ------------------------------------------------------------------------------------------------------------------------------------------------
# get_multi_dimensional_assessment
# ------------------------------------------------------------------------------------------------------------------------------------------------
def get_multi_dimensional_assessment(
    model,
    input_features,
    high_quality_stats,
    low_quality_benchmarks,
    historical_quality_scores,
    mean_y_train,
    std_dev_residuals,
    perturbation_percentage
):
    input_df = input_features.to_frame().T

    predicted_quality = model.predict(input_df)[0]

    quality_percentile = calculate_quality_percentile(predicted_quality, historical_quality_scores)

    lower_bound_expected_quality = mean_y_train - std_dev_residuals
    upper_bound_expected_quality = mean_y_train + std_dev_residuals
    expected_quality_range = f"[{lower_bound_expected_quality:.2f}, {upper_bound_expected_quality:.2f}]"

    maturity_index, maturity_status = calculate_maturity_index(
        input_features[['alcohol', 'pH', 'residual sugar']],
        high_quality_stats
    )

    risk_percentage, risk_severity = calculate_risk_degradation_index(
        input_features[['volatile acidity', 'chlorides', 'total sulfur dioxide', 'pH']],
        low_quality_benchmarks
    )

    stability_score, stability_interpretation = calculate_stability_score(
        model, input_features, perturbation_percentage
    )

    assessment_results = {
        'predicted_quality': float(predicted_quality),
        'quality_percentile': float(quality_percentile),
        'expected_quality_range': expected_quality_range,
        'maturity_index': float(maturity_index),
        'maturity_status': maturity_status,
        'risk_percentage': float(risk_percentage),
        'risk_severity': risk_severity,
        'stability_score': float(stability_score),
        'stability_interpretation': stability_interpretation
    }

    return assessment_results


# ------------------------------------------------------------------------------------------------------------------------------------------------
# full_wine_assessment
# ------------------------------------------------------------------------------------------------------------------------------------------------
def full_wine_assessment(
    model,
    input_features,
    high_quality_stats,
    low_quality_benchmarks,
    historical_quality_scores,
    mean_y_train,
    std_dev_residuals,
    perturbation_percentage,
    sorted_importance_df
):
    assessment_results = get_multi_dimensional_assessment(
        model=model,
        input_features=input_features,
        high_quality_stats=high_quality_stats,
        low_quality_benchmarks=low_quality_benchmarks,
        historical_quality_scores=historical_quality_scores,
        mean_y_train=mean_y_train,
        std_dev_residuals=std_dev_residuals,
        perturbation_percentage=perturbation_percentage
    )

    chemical_property_recommendations = recommend_chemical_properties(
        input_features=input_features,
        predicted_quality=assessment_results['predicted_quality'],
        high_quality_stats=high_quality_stats,
        low_quality_benchmarks=low_quality_benchmarks,
        sorted_importance_df=sorted_importance_df
    )

    overall_decision = make_overall_decision(assessment_results)

    final_assessment = {
        'multi_dimensional_metrics': assessment_results,
        'chemical_property_recommendations': chemical_property_recommendations,
        'overall_decision': overall_decision
    }

    return final_assessment
