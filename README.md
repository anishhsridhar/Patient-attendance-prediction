# Medical Appointment No-Show Prediction & Demand Forecasting

 A Machine Learning based healthcare analytics project that predicts:

- Patient no-show probability
- Future appointment demand forecasting

Built using:
- Python
- Scikit-learn
- XGBoost
- Streamlit

---

#  Project Overview

Missed medical appointments create major operational problems in hospitals and clinics, including:

- Wasted doctor time
- Increased waiting periods
- Poor resource allocation
- Revenue loss.

This project helps solve those problems by using Machine Learning to:

1. Predict patients who are likely to miss appointments.
2. Forecast future appointment demand.

---

# Features

## No-Show Prediction
Predicts whether a patient is likely to miss an appointment or not.

### Models Used:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### Final Selected Model:
- Random Forest Classifier

---

## Demand Forecasting
Forecasts demand for future daily appointments.

### Models Used:
- Random Forest Regressor
- XGBoost Regressor
- ARIMA

### Final Selected Model
- XGBoost Regressor

---

# Dataset Features

Some important features used in this project:

- Age
- Gender
- Specialty
- Health conditions like Hypertension, Diabetes and Alcoholism
- SMS Reminder
- Appointment Day
- Place / City
- Lag Features
- Rolling Average Features

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Matplotlib | Visualization |
| Scikit-learn | Machine Learning |
| XGBoost | Advanced ML Models |
| Statsmodels | ARIMA Forecasting |
| Streamlit | Web Application |

---

# Machine Learning Workflow

## 1. Data Cleaning
- Missing value handling
- Data formatting

## 2. Feature Engineering
- One-hot encoding
- Target encoding
- Lag features
- Rolling averages

## 3. Classification Pipeline
- Train/test split
- SMOTE balancing
- Model comparison
- Hyperparameter tuning

## 4. Forecasting Pipeline
- Daily aggregation
- Chronological split
- Time-series feature creation
- Forecast model comparison

---

# Evaluation Metrics

## Classification Metrics
- F1 Score
- ROC-AUC
- Precision
- Recall

## Forecasting Metrics
- MAE
- MAPE
- R² Score


---

#  Streamlit Dashboard

The project includes a Streamlit web application with:

- No-show Prediction
- Demand Forecasting
- Analytical Dashboard
- Business Insights

---

# How to Run

## 1. Clone Repository

```bash
git clone https://github.com/anishhsridhar/Patient-attendance-prediction.git
```

---

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 3. Run Streamlit App

```bash
streamlit run app.py
```

---

# Final Results

## Classification
| Metric | Score |
|---|---|
| F1 Score | 0.72 |
| ROC-AUC | 0.89 |

---

## Forecasting
| Metric | Score |
|---|---|
| MAPE | 19.15 % |
| R² Score | 0.92 |

---

# Future Improvements

- Real-time appointment integration
- Deep learning forecasting
- Patient notification automation
- Multi-hospital analytics dashboard

---
Developed as a Healthcare Analytics & Machine Learning project using Python.
