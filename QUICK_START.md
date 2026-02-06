# Quick Start Guide - Loan Prediction UI

## Step 1: Save the Model

First, ensure the model files are saved:

```bash
python3 save_model.py
```

This will create:
- `models/random_forest_model.pkl`
- `models/scaler.pkl`
- `models/feature_columns.pkl`

## Step 2: Install Next.js Dependencies

```bash
cd loan-prediction-ui
npm install
```

## Step 3: Run the Development Server

```bash
npm run dev
```

## Step 4: Open in Browser

Navigate to: http://localhost:3000

## Testing the Model

Fill in the form with sample values:

**Example Approved Loan (values in ₱):**
- Number of Dependents: 0
- Annual Income: 1000000
- Loan Amount: 500000
- Loan Term: 24
- CIBIL Score: 750
- Residential Assets: 2000000
- Commercial Assets: 500000
- Luxury Assets: 300000
- Bank Assets: 1000000
- Education: Graduate (1)
- Self Employed: No (0)

**Example Rejected Loan (values in ₱):**
- Number of Dependents: 3
- Annual Income: 300000
- Loan Amount: 800000
- Loan Term: 12
- CIBIL Score: 450
- Residential Assets: 100000
- Commercial Assets: 0
- Luxury Assets: 0
- Bank Assets: 50000
- Education: Not Graduate (0)
- Self Employed: Yes (1)

## Troubleshooting

### Python script not found
- Make sure `predict.py` is in the project root directory
- Check that Python 3 is installed: `python3 --version`

### Model files not found
- Run `python3 save_model.py` from the project root
- Verify the `models/` directory exists with all three `.pkl` files

### Port already in use
- Change the port: `npm run dev -- -p 3001`
- Or kill the process using port 3000

### Module not found errors
- Make sure you're in the `loan-prediction-ui` directory when running `npm install`
- Delete `node_modules` and `package-lock.json`, then run `npm install` again
