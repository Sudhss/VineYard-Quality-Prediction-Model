"""
main.py
-------
Entry point for training the Vineyard Quality Assessment ML pipeline.
Run this once before launching the Streamlit app.

Usage:
    python main.py
"""

import sys
import os
import logging

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import get_splits
from model_engine import (
    train_model,
    evaluate_model,
    get_feature_importance,
    save_artifacts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Vineyard Quality Assessment — Training Pipeline")
    logger.info("=" * 60)

    # Step 1: Load and preprocess data
    logger.info("Step 1/4 — Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, quality_array, scaler = get_splits(
        csv_path="data/wine_quality.csv"
    )

    # Step 2: Train model with hyperparameter tuning
    logger.info("Step 2/4 — Training XGBoost model with RandomizedSearchCV...")
    model = train_model(X_train, y_train, n_iter=40, cv=5)

    # Step 3: Evaluate
    logger.info("Step 3/4 — Evaluating model performance...")
    metrics = evaluate_model(model, X_test, y_test)
    logger.info("Final Metrics → MAE: %.4f | RMSE: %.4f | R2: %.4f",
                metrics["MAE"], metrics["RMSE"], metrics["R2"])

    # Step 4: Save artifacts
    logger.info("Step 4/4 — Saving model and artifacts...")
    save_artifacts(model, scaler, quality_array, metrics)

    logger.info("=" * 60)
    logger.info("Training complete. Run: streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
