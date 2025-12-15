import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Artifact builder function
# ------------------------------------------------------------------------------------------------------------------------------------------------
def build_and_save_artifacts(
    model,
    X_train,
    y_train,
    df,
    model_dir="model"
):
    os.makedirs(model_dir, exist_ok=True)

    # -----------------------------
    # Feature importance
    # -----------------------------
    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    # -----------------------------
    # Residual statistics
    # -----------------------------
    y_train_pred = model.predict(X_train)
    residuals = y_train - y_train_pred

    mean_y_train = float(np.mean(y_train))
    std_dev_residuals = float(np.std(residuals))

    # -----------------------------
    # Quality benchmarks
    # -----------------------------
    high_quality_stats = (
        df[df["quality"] >= 7.2]
        [["alcohol", "pH", "residual sugar"]]
        .describe()
        .loc[["mean", "std"]]
    )

    low_quality_benchmarks = (
        df[df["quality"] <= 5.5]
        .drop(columns=["quality"])
        .agg(["mean", "std"])
    )

    # -----------------------------
    # Save artifacts
    # -----------------------------
    artifacts = {
        "sorted_importance_df": importance,
        "high_quality_stats": high_quality_stats,
        "low_quality_benchmarks": low_quality_benchmarks,
        "historical_quality_scores": y_train.values,
        "mean_y_train": mean_y_train,
        "std_dev_residuals": std_dev_residuals
    }

    joblib.dump(artifacts, os.path.join(model_dir, "artifacts.joblib"))
    print("Artifacts saved successfully.")


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Traing Pipeline - Run this script to train and save the model + artifacts
# ------------------------------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("data/wine_quality.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("Random Forest trained.")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/random_forest_model.joblib")
print("Model saved.")

build_and_save_artifacts(
    model=model,
    X_train=X_train,
    y_train=y_train,
    df=df,
    model_dir="model"
)
