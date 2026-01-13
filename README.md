# 🎓 Engineering College Placement Prediction & Analysis System

A machine learning–based web application that predicts student placement outcomes
and provides analytics for Training & Placement (T&P) teams.

---

## 🚀 Features

### 👨‍🎓 Student Prediction Module
- Predicts placement probability using a trained ML model
- Accepts academic, technical, and aptitude inputs
- Displays prediction confidence and performance profile

### 🏢 T&P Analytics Dashboard
- Upload placement datasets (CSV)
- Visualize placement distribution and trends
- Analyze placement by branch, CGPA, and project experience

---

## 🧠 Machine Learning Approach
- Algorithm: **Random Forest Classifier**
- Feature Engineering: One-Hot Encoding
- Probability-based predictions using `predict_proba`
- Model persistence using Joblib

---

## 🛠️ Tech Stack
- Python 3.11
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

---

## 📂 Project Structure
placement_prediction_app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│ └── placement_cleaned.csv
│
├── model/
│ ├── rf_placement_model.joblib
│ └── feature_columns.json

---

## ▶️ How to Run Locally
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

