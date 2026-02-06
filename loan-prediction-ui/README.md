# Loan Approval Prediction UI

A Next.js frontend application for testing the Loan Approval Prediction Random Forest model.

## Features

- Clean, modern UI with gradient design
- Form with all 17 required features
- Real-time loan approval prediction
- Displays prediction result with confidence scores
- Automatic calculation of derived fields (ratios, monthly values)

## Setup

1. **Install dependencies:**
```bash
cd loan-prediction-ui
npm install
```

2. **Make sure the model files exist in the parent directory:**
   - `models/random_forest_model.pkl`
   - `models/scaler.pkl`
   - `models/feature_columns.pkl`

   If they don't exist, run from the project root:
```bash
python3 save_model.py
```

3. **Run the development server:**
```bash
npm run dev
```

4. **Open [http://localhost:3000](http://localhost:3000) in your browser**

## Model Information

- **Model Type:** Random Forest Classifier
- **Accuracy:** 99.88%
- **F1-Score:** 99.84%
- **ROC-AUC:** 100%

## API Endpoint

The app uses `/api/predict` endpoint which:
1. Receives form data
2. Calls Python script (`predict.py`) to load model and make prediction
3. Returns prediction result with confidence scores

## Notes

- The Python prediction script (`predict.py`) must be in the parent directory
- Ensure Python 3 is installed with required packages (scikit-learn, joblib, numpy)
- The model expects all 17 features in the correct order
- Derived fields (ratios, monthly values) are calculated automatically in the frontend

## Example Input Values

All monetary values are in **Philippine Pesos (₱)**.

- **CIBIL Score:** 750+ (good), 600-750 (fair), <600 (poor)
- **Loan Amount:** Typically 1-10x annual income
- **Loan Term:** Usually 12-60 months
- **Education:** 0 = Not Graduate, 1 = Graduate
- **Self Employed:** 0 = No, 1 = Yes
