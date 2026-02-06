# Loan Approval Prediction

**Using Classical and Modern Machine Learning Techniques**

A machine learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant features such as income, credit score, assets, and financial ratios.

*Final project for a Machine Learning class*

---

## Overview

| Aspect | Details |
|--------|---------|
| **Task** | Binary classification (Approved vs. Rejected) |
| **Dataset** | Kaggle Loan Approval Dataset — 4,269 applications |
| **Features** | 17 (income, loan amount, CIBIL score, assets, ratios, etc.) |
| **Best Model** | Random Forest — 99.88% accuracy, 0.9984 F1-Score |
| **Top Predictor** | CIBIL score (~67% importance) |

---

## Quick Start

### 1. Setup

```bash
git clone <repository-url>
cd loan-approval-prediction
pip install -r requirements.txt
```

### 2. Run the Pipeline

**Step 2 — Data Preprocessing & EDA:**
```bash
jupyter notebook Step2_Data_Preprocessing.ipynb
```
Run all cells to produce the preprocessed dataset and EDA visualizations.

**Step 3 — Model Development:**
```bash
jupyter notebook Step3_Model_Development.ipynb
```
Run all cells to train 8+ models, compare performance, and generate charts.

### 3. Regenerate Step 2 Visualizations (optional)

```bash
python export_visualizations.py
```

---

## Project Structure

```
loan-approval-prediction/
├── data/                              # Datasets & results
│   ├── loan_approval_dataset.csv      # Raw data
│   ├── loan_approval_dataset_preprocessed.csv
│   ├── model_results.csv              # Model metrics
│   └── feature_importance.csv
├── presentation_images/               # Charts for slides (17 images)
├── Step2_Data_Preprocessing.ipynb     # EDA & preprocessing
├── Step3_Model_Development.ipynb      # Model training & comparison
├── export_visualizations.py           # Regenerate Step 2 charts
├── requirements.txt
├── Loan Approval Prediction.pdf       # Presentation slides
├── PROJECT_OVERVIEW.md                # Detailed project guide
└── README_Step2_Submission.md         # Step 2 submission quick start
```

---

## Models Trained

| Category | Models |
|----------|--------|
| **Classical ML** | Logistic Regression, L1/L2/ElasticNet, PCA, Cross-validation |
| **Modern ML** | SVM (Linear, RBF, Polynomial), Random Forest, XGBoost, Neural Network |

---

## Key Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| **Random Forest** | **99.88%** | **0.9984** | **1.0000** |
| XGBoost | 99.77% | 0.9969 | 1.0000 |
| Neural Network | 98.24% | 0.9765 | 0.9970 |
| SVM (Tuned RBF) | 95.78% | 0.9436 | 0.9913 |
| Logistic Regression | 94.50% | 0.9280 | 0.9778 |

---

## Dependencies

- pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
- jupyter
- tensorflow
- xgboost

See `requirements.txt` for versions.

---

## Documentation

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** — Full project description, dataset details, workflow, and status
- **Step2_Presentation_Guide.md** — How to build Step 2 slides
- **Step3_5_Presentation_Guide.md** — Slide content for Steps 3–5 (model development, evaluation, conclusion)

---

## License

[Add your license here if applicable]
