'use client'

import { useState } from 'react'

// Store number inputs as strings so typing works naturally (no leading zero issue)
interface FormData {
  no_of_dependents: string
  income_annum: string
  loan_amount: string
  loan_term: string
  cibil_score: string
  residential_assets_value: string
  commercial_assets_value: string
  luxury_assets_value: string
  bank_asset_value: string
  education_encoded: number
  self_employed_encoded: number
  total_assets_value: number
  loan_to_income_ratio: number
  assets_to_loan_ratio: number
  monthly_income: number
  monthly_loan_payment: number
  debt_to_income_ratio: number
}

interface PredictionResult {
  prediction: string
  probability: number
  confidence: number
}

export default function Home() {
  const [formData, setFormData] = useState<FormData>({
    no_of_dependents: '',
    income_annum: '',
    loan_amount: '',
    loan_term: '',
    cibil_score: '',
    residential_assets_value: '',
    commercial_assets_value: '',
    luxury_assets_value: '',
    bank_asset_value: '',
    education_encoded: 0,
    self_employed_encoded: 0,
    total_assets_value: 0,
    loan_to_income_ratio: 0,
    assets_to_loan_ratio: 0,
    monthly_income: 0,
    monthly_loan_payment: 0,
    debt_to_income_ratio: 0,
  })

  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Parse string to number, return 0 if empty or invalid
  const toNum = (val: string | number): number => {
    if (typeof val === 'number') return val
    const parsed = parseFloat(val)
    return isNaN(parsed) ? 0 : parsed
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    const field = name as keyof FormData
    // Number inputs: store as string for natural typing (no 0123 issue)
    const isNumberInput = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
      'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'].includes(name)
    setFormData(prev => ({
      ...prev,
      [field]: isNumberInput ? value : (parseFloat(value) || 0)
    }))
  }

  const calculateDerivedFields = () => {
    const income = toNum(formData.income_annum)
    const loanAmount = toNum(formData.loan_amount)
    const loanTerm = toNum(formData.loan_term)
    
    // Calculate monthly income
    const monthlyIncome = income / 12
    
    // Calculate monthly loan payment (simple calculation)
    const monthlyLoanPayment = loanTerm > 0 ? loanAmount / loanTerm : 0
    
    // Calculate total assets
    const totalAssets = toNum(formData.residential_assets_value) + 
                       toNum(formData.commercial_assets_value) + 
                       toNum(formData.luxury_assets_value) + 
                       toNum(formData.bank_asset_value)
    
    // Calculate ratios
    const loanToIncomeRatio = income > 0 ? loanAmount / income : 0
    const assetsToLoanRatio = loanAmount > 0 ? totalAssets / loanAmount : 0
    const debtToIncomeRatio = income > 0 ? monthlyLoanPayment * 12 / income : 0
    
    return {
      monthly_income: monthlyIncome,
      monthly_loan_payment: monthlyLoanPayment,
      total_assets_value: totalAssets,
      loan_to_income_ratio: loanToIncomeRatio,
      assets_to_loan_ratio: assetsToLoanRatio,
      debt_to_income_ratio: debtToIncomeRatio,
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Calculate derived fields
      const derived = calculateDerivedFields()
      
      // Prepare submission data (convert string inputs to numbers)
      const submissionData = {
        no_of_dependents: toNum(formData.no_of_dependents),
        income_annum: toNum(formData.income_annum),
        loan_amount: toNum(formData.loan_amount),
        loan_term: toNum(formData.loan_term),
        cibil_score: toNum(formData.cibil_score),
        residential_assets_value: toNum(formData.residential_assets_value),
        commercial_assets_value: toNum(formData.commercial_assets_value),
        luxury_assets_value: toNum(formData.luxury_assets_value),
        bank_asset_value: toNum(formData.bank_asset_value),
        education_encoded: formData.education_encoded,
        self_employed_encoded: formData.self_employed_encoded,
        ...derived
      }

      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(submissionData),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to make prediction')
      }

      const predictionResult = await response.json()
      setResult(predictionResult)
    } catch (err: any) {
      setError(err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>🏦 Loan Approval Prediction</h1>
      <p className="subtitle">
        Enter applicant details to predict loan approval using our Random Forest ML model
      </p>

      <div className="info-box">
        <p><strong>Model Performance:</strong> 99.88% Accuracy, 99.84% F1-Score, 100% ROC-AUC</p>
        <p><strong>Top Predictors:</strong> CIBIL Score (81.9%), Debt-to-Income Ratio (4.7%), Loan-to-Income Ratio (3.0%)</p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          <div className="form-group">
            <label htmlFor="no_of_dependents">Number of Dependents</label>
            <input
              type="number"
              id="no_of_dependents"
              name="no_of_dependents"
              value={formData.no_of_dependents}
              onChange={handleChange}
              placeholder="0"
              min="0"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="income_annum">Annual Income (₱)</label>
            <input
              type="number"
              id="income_annum"
              name="income_annum"
              value={formData.income_annum}
              onChange={handleChange}
              placeholder="0"
              min="0"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="loan_amount">Loan Amount (₱)</label>
            <input
              type="number"
              id="loan_amount"
              name="loan_amount"
              value={formData.loan_amount}
              onChange={handleChange}
              placeholder="0"
              min="0"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="loan_term">Loan Term (months)</label>
            <input
              type="number"
              id="loan_term"
              name="loan_term"
              value={formData.loan_term}
              onChange={handleChange}
              placeholder="12"
              min="1"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="cibil_score">CIBIL Score (300-900)</label>
            <input
              type="number"
              id="cibil_score"
              name="cibil_score"
              value={formData.cibil_score}
              onChange={handleChange}
              placeholder="300"
              min="300"
              max="900"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="residential_assets_value">Residential Assets Value (₱)</label>
            <input
              type="number"
              id="residential_assets_value"
              name="residential_assets_value"
              value={formData.residential_assets_value}
              onChange={handleChange}
              placeholder="0"
              min="0"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="commercial_assets_value">Commercial Assets Value (₱)</label>
            <input
              type="number"
              id="commercial_assets_value"
              name="commercial_assets_value"
              value={formData.commercial_assets_value}
              onChange={handleChange}
              placeholder="0"
              min="0"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="luxury_assets_value">Luxury Assets Value (₱)</label>
            <input
              type="number"
              id="luxury_assets_value"
              name="luxury_assets_value"
              value={formData.luxury_assets_value}
              onChange={handleChange}
              placeholder="0"
              min="0"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="bank_asset_value">Bank Asset Value (₱)</label>
            <input
              type="number"
              id="bank_asset_value"
              name="bank_asset_value"
              value={formData.bank_asset_value}
              onChange={handleChange}
              placeholder="0"
              min="0"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="education_encoded">Education</label>
            <select
              id="education_encoded"
              name="education_encoded"
              value={formData.education_encoded}
              onChange={handleChange}
              required
            >
              <option value="0">Not Graduate</option>
              <option value="1">Graduate</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="self_employed_encoded">Self Employed</label>
            <select
              id="self_employed_encoded"
              name="self_employed_encoded"
              value={formData.self_employed_encoded}
              onChange={handleChange}
              required
            >
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </div>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict Loan Approval'}
        </button>
      </form>

      {loading && (
        <div className="loading">
          ⏳ Processing your request...
        </div>
      )}

      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className={`result-container ${result.prediction === 'Approved' ? 'result-approved' : 'result-rejected'}`}>
          <div className="result-title">
            {result.prediction === 'Approved' ? '✅ Loan Approved!' : '❌ Loan Rejected'}
          </div>
          <div className="result-details">
            <div className="result-item">
              <span>Prediction:</span>
              <span><strong>{result.prediction}</strong></span>
            </div>
            <div className="result-item">
              <span>Rejection Probability:</span>
              <span><strong>{(result.probability * 100).toFixed(2)}%</strong></span>
            </div>
            <div className="result-item">
              <span>Confidence:</span>
              <span><strong>{(result.confidence * 100).toFixed(2)}%</strong></span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
