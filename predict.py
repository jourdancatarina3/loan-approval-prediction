"""
Python script to make predictions using the saved Random Forest model.
This script is called by the Next.js API route.
"""

import sys
import json
import joblib
import numpy as np
from pathlib import Path

# Get the input file path from command line argument
if len(sys.argv) < 2:
    print(json.dumps({"error": "No input file provided"}))
    sys.exit(1)

input_file = sys.argv[1]

try:
    # Load the feature values from JSON file
    with open(input_file, 'r') as f:
        feature_values = json.load(f)
    
    # Convert to numpy array and reshape
    features = np.array(feature_values).reshape(1, -1)
    
    # Load the model, scaler, and feature columns
    models_dir = Path(__file__).parent / 'models'
    
    scaler = joblib.load(models_dir / 'scaler.pkl')
    model = joblib.load(models_dir / 'random_forest_model.pkl')
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get probability of rejection (class 1)
    rejection_prob = probabilities[1]
    approval_prob = probabilities[0]
    
    # Determine confidence
    confidence = max(approval_prob, rejection_prob)
    
    # Return result as JSON
    result = {
        "prediction": int(prediction),
        "probability": float(rejection_prob),
        "confidence": float(confidence)
    }
    
    print(json.dumps(result))
    
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
