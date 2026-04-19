"""
model_engine.py
---------------
Trains an XGBRegressor on the Wine Quality dataset.
Performs RandomizedSearchCV hyperparameter tuning.
Saves the trained model and supporting artifacts via joblib.
Evaluates with MAE, RMSE, and R² metrics.
"""

import os
import logging
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "model/xgboost_model.joblib"
ARTIFACTS_PATH = "model/artifacts.joblib"

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

# Hyperparameter search space for RandomizedSearchCV
PARAM_DIST = {
    "n_estimators": [200, 300, 400, 500, 600],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 4, 5],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1, 0.5],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],
}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 40,
    cv: int = 5,
    random_state: int = 42,
) -> XGBRegressor:

    os.makedirs("model", exist_ok=True)

    base_model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    logger.info("Starting RandomizedSearchCV — %d iterations, %d-fold CV", n_iter, cv)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV RMSE: %.4f", np.sqrt(-search.best_score_))

    return best_model


def evaluate_model(
    model: XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

    logger.info("Evaluation — MAE: %.4f | RMSE: %.4f | R2: %.4f", mae, rmse, r2)
    return metrics


def get_feature_importance(model: XGBRegressor) -> dict:
    """
    Extract normalized feature importances from the trained XGBoost model.
    Returns a dict mapping feature name → importance score (0–1).
    """
    raw_importance = model.feature_importances_
    total = raw_importance.sum()
    normalized = raw_importance / total if total > 0 else raw_importance

    importance_dict = {
        name: round(float(score), 4)
        for name, score in zip(FEATURE_NAMES, normalized)
    }

    # Sort descending by importance
    importance_dict = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )
    return importance_dict


def save_artifacts(
    model: XGBRegressor,
    scaler,
    quality_array: np.ndarray,
    metrics: dict,
) -> None:
    """
    Persist the trained model and all supporting artifacts to disk.
    - model/xgboost_model.joblib : the XGBRegressor
    - model/artifacts.joblib     : scaler, quality distribution, metrics
    """
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    artifacts = {
        "scaler": scaler,
        "quality_distribution": quality_array,
        "metrics": metrics,
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)
    logger.info("Artifacts saved to %s", ARTIFACTS_PATH)


def load_model_and_artifacts():
    """
    Load the saved model and artifacts from disk.
    Returns (model, artifacts_dict).
    Raises FileNotFoundError if not found.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError(
            f"Model files not found. Run main.py first to train the model.\n"
            f"Expected: {MODEL_PATH} and {ARTIFACTS_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)
    logger.info("Model and artifacts loaded successfully.")
    return model, artifacts
