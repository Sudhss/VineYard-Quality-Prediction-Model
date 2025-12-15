import json
import joblib
import pandas as pd
import os

from src.assessment_engine import full_wine_assessment

# ------------------------------------
# Load model + artifacts
# ------------------------------------
model = joblib.load("model/random_forest_model.joblib")
artifacts = joblib.load("model/artifacts.joblib")

# ------------------------------------
# Example user input (later comes from React)
# ------------------------------------

user_input = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

input_features = pd.Series(user_input)

# ------------------------------------
# Run assessment
# ------------------------------------
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

# ------------------------------------
# WRITE JSON TO FILE  
# ------------------------------------
os.makedirs("output", exist_ok=True)

with open("output/assessment_result.json", "w") as f:
    json.dump(result, f, indent=4)

print("JSON generated at output/assessment_result.json")
