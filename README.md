# Heart Disease Prediction
> A machine learning project to predict the presence of heart disease using clinical patient data.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow.svg)

---

## Overview

This project builds and evaluates multiple machine learning models to predict whether a patient has heart disease based on 13 clinical features. The best model achieves **83.6% accuracy** and a **0.91 ROC-AUC score**.

---

## Dataset

- **Source**: [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-uci) via Kaggle
- **Rows**: 303 patients
- **Features**: 13 clinical features + 1 target column
- **Target**: 0 = No heart disease, 1 = Heart disease

### Features

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Sex (1 = Male, 0 = Female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = True) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (1 = Yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels coloured by fluoroscopy |
| thal | Thalassemia type (0–3) |

---

## Project Structure

```
Heart_Disease_Prediction/
│
├── heart.csv                        # Dataset
├── Heart_Disease_Prediction.ipynb   # Main notebook
├── heart_disease_model.pkl          # Saved Random Forest model
├── scaler.pkl                       # Saved StandardScaler
└── README.md                        # Project documentation
```

---

## Methodology

```
Data Loading → EDA → Cleaning → Preprocessing → Model Training → Evaluation → Tuning
```

1. **Exploratory Data Analysis** — distributions, correlation heatmap, target balance
2. **Data Cleaning** — verified zero missing values, confirmed binary target
3. **Preprocessing** — StandardScaler on continuous features, 80/20 train/test split
4. **Model Training** — Logistic Regression, Random Forest, XGBoost
5. **Evaluation** — accuracy, ROC-AUC, confusion matrix, classification report
6. **Hyperparameter Tuning** — GridSearchCV on Random Forest (540 combinations)
7. **Live Prediction** — interactive function to predict on new patient data
8. **Model Export** — saved with joblib for reuse

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **Random Forest** | **83.6%** | **0.91** |
| Logistic Regression | 80.3% | 0.87 |
| XGBoost | 80.3% | 0.86 |

**Best model: Random Forest** with the following tuned parameters:
- `max_depth`: 5
- `min_samples_leaf`: 4
- `min_samples_split`: 2
- `n_estimators`: 200

### Key Findings

- `cp` (chest pain type) is the strongest positive predictor (+0.43 correlation)
- `thalach` (max heart rate) is the second strongest (+0.42)
- `exang` (exercise angina) and `oldpeak` are the strongest negative predictors
- All models achieved high recall (91–97%) for detecting disease — critical for medical use

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Nehu2021/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

---

## Usage

```bash
# Launch Jupyter Notebook
jupyter notebook

# Open Heart_Disease_Prediction.ipynb and run all cells
```

To predict on a new patient:

```python
predict_patient(
    age=55, sex=1, cp=2, trestbps=130, chol=250,
    fbs=0, restecg=1, thalach=150, exang=0,
    oldpeak=1.5, slope=1, ca=0, thal=2
)
```

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
jupyter
```

---

## Author

**Neha Khatri**  
[GitHub](https://github.com/neha-khatry). [LinkedIn](https://www.linkedin.com/in/neha-k-1a5917335) 

---

## License

This project is open source and available under the [MIT License](LICENSE).
