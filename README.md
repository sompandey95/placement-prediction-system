# Engineering College Placement Prediction System

A major project for predicting engineering student placement outcomes using machine learning, with explainable AI and analytics dashboard.

## Features
- Multi-model comparison (Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM, KNN)
- SMOTE for class imbalance handling
- Hyperparameter tuning with RandomizedSearchCV
- SHAP explainability per prediction
- Student prediction with probability gauge and radar chart
- T&P Analytics Dashboard with correlation heatmap and at-risk student table
- Batch prediction with CSV download
- PDF report generation per student

## Project Structure
placement-prediction-system/
├── data/                        # Dataset
├── model/                       # Saved model artifacts
├── src/
│   ├── preprocess.py            # Data loading, encoding, SMOTE
│   ├── predict.py               # Inference logic
│   ├── evaluate.py              # Model comparison loader
│   └── report_gen.py            # PDF report generator
├── views/
│   ├── student.py               # Student prediction page
│   └── dashboard.py             # T&P dashboard page
├── app.py                       # Streamlit entry point
├── train.py                     # Model training pipeline
└── config.yaml                  # Configuration

## Setup
```bash
pip install -r requirements.txt
```

## Train the Model
```bash
python train.py
```

## Run the App
```bash
streamlit run app.py
```

## Model Results
| Model | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | 82.5% | 0.817 | 0.842 |
| Random Forest | 82.5% | 0.809 | 0.854 |
| XGBoost | 77.5% | 0.765 | 0.810 |
| Gradient Boosting | 80.0% | 0.800 | 0.845 |
| SVM | 62.5% | 0.639 | 0.313 |
| KNN | 55.0% | 0.568 | 0.573 |

## Tech Stack

Python, Streamlit, scikit-learn, XGBoost, SHAP, imbalanced-learn, Plotly, fpdf2
