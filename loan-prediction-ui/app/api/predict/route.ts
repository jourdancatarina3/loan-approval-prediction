import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import { execSync } from 'child_process';
import fs from 'fs';
import os from 'os';

export async function POST(request: NextRequest) {
  let tempFile: string | null = null;
  
  try {
    const body = await request.json();
    
    // Validate input
    const requiredFields = [
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
    ];

    // Check if all fields are present
    for (const field of requiredFields) {
      if (body[field] === undefined || body[field] === null) {
        return NextResponse.json(
          { error: `Missing required field: ${field}` },
          { status: 400 }
        );
      }
    }

    // Prepare data for Python script
    const featureValues = requiredFields.map(field => body[field]);
    
    // Get project root (parent directory)
    const projectRoot = path.join(process.cwd(), '..');
    const pythonScript = path.join(projectRoot, 'predict.py');
    
    // Create a temporary JSON file with the input data
    tempFile = path.join(os.tmpdir(), `loan_pred_${Date.now()}.json`);
    fs.writeFileSync(tempFile, JSON.stringify(featureValues));

    // Call Python script to make prediction
    const result = execSync(
      `python3 "${pythonScript}" "${tempFile}"`,
      { 
        cwd: projectRoot,
        encoding: 'utf-8',
        maxBuffer: 1024 * 1024 * 10 // 10MB buffer
      }
    );
    
    const prediction = JSON.parse(result.trim());
    
    if (prediction.error) {
      return NextResponse.json(
        { error: prediction.error },
        { status: 500 }
      );
    }
    
    return NextResponse.json({
      prediction: prediction.prediction === 0 ? 'Approved' : 'Rejected',
      probability: prediction.probability,
      confidence: prediction.confidence
    });
  } catch (error: any) {
    console.error('Prediction error:', error);
    return NextResponse.json(
      { error: 'Failed to make prediction', details: error.message },
      { status: 500 }
    );
  } finally {
    // Clean up temp file if it exists
    if (tempFile && fs.existsSync(tempFile)) {
      try {
        fs.unlinkSync(tempFile);
      } catch (e) {
        console.error('Error cleaning up temp file:', e);
      }
    }
  }
}
