# Vineyard Quality Assessment System

Production-grade ML system for predicting wine quality from chemical properties,
powered by XGBoost with full hyperparameter tuning and a premium Streamlit dashboard.

---

## Architecture

```
Vineyard ML Model/
├── data/                     # UCI Wine Quality dataset (auto-downloaded)
│   └── wine_quality.csv
├── model/                    # Trained model artifacts
│   ├── xgboost_model.joblib
│   └── artifacts.joblib
├── src/
│   ├── data_loader.py        # Dataset download, preprocessing, splitting
│   ├── model_engine.py       # XGBoost training, tuning, evaluation, save/load
│   └── assessment_engine.py  # Inference, percentile, maturity, risk, stability
├── output/
│   └── assessment_result.json
├── main.py                   # Training pipeline entry point
├── app.py                    # Streamlit frontend
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python main.py
```
This will:
- Download the UCI Wine Quality dataset (red + white combined)
- Preprocess and split the data
- Run RandomizedSearchCV over XGBoost hyperparameters
- Evaluate on the test set (MAE, RMSE, R2)
- Save `model/xgboost_model.joblib` and `model/artifacts.joblib`

### 3. Launch the app
```bash
streamlit run app.py
```

---

## ML Details

| Component | Detail |
|---|---|
| Algorithm | XGBRegressor |
| Tuning | RandomizedSearchCV (40 iterations, 5-fold CV) |
| Metric | neg_mean_squared_error |
| Preprocessing | StandardScaler |
| Dataset | UCI Red + White Wine Quality (combined) |
| Output | Regression score on 3–9 scale |

### Hyperparameters tuned
- `n_estimators`: [200, 300, 400, 500, 600]
- `max_depth`: [3, 4, 5, 6, 7]
- `learning_rate`: [0.01, 0.03, 0.05, 0.07, 0.1, 0.15]
- `subsample`: [0.6 – 1.0]
- `colsample_bytree`: [0.6 – 1.0]
- `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`

---

## Assessment Output

```json
{
  "predicted_quality": 5.02,
  "quality_percentile": 47.1,
  "maturity_index": 68.3,
  "maturity_status": "Peak Maturity",
  "risk_percentage": 72.5,
  "risk_severity": "High",
  "stability": "Stable",
  "recommendations": [
    "Reduce volatile acidity from 0.70 toward optimal range [0.20 – 0.50]",
    "Increase alcohol from 9.40 toward optimal range [10.00 – 13.00]"
  ],
  "feature_importance": { ... }
}
```

---

## Dataset

UCI Machine Learning Repository — Wine Quality Data Set  
P. Cortez et al., 2009. Using data mining for wine quality assessment.  
Red wine: 1,599 samples | White wine: 4,898 samples | Combined: 6,497 samples
