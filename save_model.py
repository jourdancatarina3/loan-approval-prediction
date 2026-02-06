"""
Script to train and save the Random Forest model for loan approval prediction.
Run this script after running the Step3_Model_Development.ipynb notebook
to ensure the model is saved for the Next.js API.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Loading preprocessed dataset...")
df = pd.read_csv('data/loan_approval_dataset_preprocessed.csv')

# Features to use for modeling (17 features)
feature_columns = [
    'no_of_dependents',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value',
    'education_encoded',
    'self_employed_encoded',
    'total_assets_value',
    'loan_to_income_ratio',
    'assets_to_loan_ratio',
    'monthly_income',
    'monthly_loan_payment',
    'debt_to_income_ratio'
]

# Define X (features) and y (target)
X = df[feature_columns]
y = df['loan_status_encoded']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE, 
    stratify=y
)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print("Training Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

# Evaluate model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Save model and scaler
print("\nSaving model and scaler...")
joblib.dump(rf, 'models/random_forest_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl')

print("Model saved successfully!")
print("Files saved:")
print("  - models/random_forest_model.pkl")
print("  - models/scaler.pkl")
print("  - models/feature_columns.pkl")
