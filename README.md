# 🍇 Vineyard Quality Assessment System 🍇

A machine learning system for predicting wine quality and providing actionable insights for vineyard management using chemical properties of wine.

## Features

- **Quality Prediction**: Predicts wine quality scores using a trained Random Forest model
- **Maturity Assessment**: Evaluates wine maturity status (Under-mature, Peak Maturity, Over-mature)
- **Risk Analysis**: Identifies potential quality degradation risks
- **Production-ready**: Structured JSON output for easy integration with web applications
- **Explainable AI**: Provides feature importance and reasoning behind predictions

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vineyard-ml-model.git
   cd vineyard-ml-model
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Install Dependencies
First, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Run the data loader to download and prepare the dataset:
```bash
python -m src.data_loader
```

### 3. Model Training
Train the model and generate artifacts:
```bash
python -m src.model_engine
```

### 4. Run the Web Interface
Start the Streamlit web application:
```bash
streamlit run app.py
```

Then open your web browser to the URL shown in the terminal (usually http://localhost:8501) to access the interactive interface.
1. Enter the values recieved from the test results of the vineyard
2. Click Test
3. You will get a detailed map of the entire quality along with neccessary steps to imrpove quality

## Project Structure

```
Vineyard ML Model/
├── data/                    # Dataset storage
│   └── wine_quality.csv     # Wine quality dataset
│
├── model/                   # Trained models and artifacts
│   ├── random_forest_model.joblib
│   └── artifacts.joblib
│
├── src/                     # Source code
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── model_engine.py      # Model training and evaluation
│   └── assessment_engine.py # Core assessment logic
│
├── output/                  # Generated assessment results
│   └── assessment_result.json
│
├── main.py                  # Application entry point
└── requirements.txt         # Python dependencies
```

## Model Performance

The model provides the following performance metrics:
- Mean Absolute Error (MAE): ~0.45
- R² Score: ~0.45

## Output Format

The assessment result includes:
- Predicted quality score
- Quality percentile
- Maturity assessment
- Risk analysis
- Feature importance
- Chemical property recommendations

Example output (simplified):
```json
{
  "predicted_quality": 6.2,
  "quality_percentile": 85.3,
  "maturity_index": 75.4,
  "maturity_status": "Peak Maturity",
  "risk_percentage": 22.1,
  "risk_severity": "Low",
  "recommendations": [
    "Slight reduction in volatile acidity could improve quality",
    "Current alcohol content is optimal for this wine profile"
  ]
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author
[Sudhanshu Shukla (@Sudhss)](https://github.com/Sudhss)

## Resources

- [UCI Machine Learning Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Joblib Documentation](https://joblib.readthedocs.io/)
