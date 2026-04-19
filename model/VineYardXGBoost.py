import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# -----------------------------
# 2. FEATURES & TARGET
# -----------------------------
X = df.drop("quality", axis=1)
y = df["quality"]

# -----------------------------
# 3. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. BEST-TUNED MODEL (NO GRIDSEARCH BS)
# -----------------------------
model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 5. PREDICTIONS
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6. METRICS
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# -----------------------------
# 8. SAVE ARTIFACTS (IMPORTANT)
# -----------------------------
artifacts = {
    "feature_names": list(X.columns),
    "model_type": "XGBoost Regressor",
    "target": "quality",
    "quality_distribution": y.values.tolist(),  # ADD THIS
    "metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }
}

import os

MODEL_DIR = r"C:\Users\shukl\Downloads\VineyardML_System\VineyardML\model"

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_model.joblib"))

joblib.dump(artifacts, os.path.join(MODEL_DIR, "artifacts.joblib"))

print("Saved inside model folder ✅")
print("Artifacts + Model saved ✅")

# -----------------------------
# 9. ACTUAL vs PREDICTED PLOT
# -----------------------------
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Actual vs Predicted Quality")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.savefig("Fig_5_1_Actual_vs_Predicted.png", dpi=300)
plt.show()

# -----------------------------
# 10. FEATURE IMPORTANCE
# -----------------------------
importance = model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importance)
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance")

plt.savefig("Fig_5_2_Feature_Importance.png", dpi=300)
plt.show()  