"""
Flask Backend API for Financial Health Prediction
Serves the trained LightGBM model via REST API
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

# Load the trained model
MODEL_PATH = '/models/financial_health_model.pkl'
model_data = None

def load_model():
    """Load the trained model on startup"""
    global model_data
    try:
        model_data = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def create_features(data):
    """
    Engineer features from input data
    Matches the feature engineering in training script
    """
    df = pd.DataFrame([data])
    
    # 1. Profit Margin
    df['profit_margin'] = np.where(
        (df['personal_income'].notna()) & 
        (df['business_expenses'].notna()) & 
        (df['personal_income'] != 0),
        (df['personal_income'] - df['business_expenses']) / df['personal_income'],
        np.nan
    )
    df['profit_margin'] = df['profit_margin'].clip(-1, 1)
    
    # 2. Financial Access Score
    financial_features = [
        'has_bank_account', 'has_loan_account', 'has_internet_banking',
        'has_debit_card', 'medical_insurance', 'funeral_insurance'
    ]
    
    def calc_access_score(row):
        score = 0
        valid = 0
        for feat in financial_features:
            val = row.get(feat)
            if pd.notna(val) and val != '':
                valid += 1
                if val in ['Yes', 'Have now', 'have now']:
                    score += 1
                elif val in ["Used to have but don't have now", 'used to have']:
                    score += 0.5
        return score / valid if valid > 0 else np.nan
    
    df['financial_access_score'] = df.apply(calc_access_score, axis=1)
    
    # 3. Income to Expenses Ratio
    df['income_expense_ratio'] = np.where(
        (df['business_expenses'].notna()) & (df['business_expenses'] > 0),
        df['personal_income'] / df['business_expenses'],
        np.nan
    )
    
    # 4. Business Age Category
    if 'business_age_months' in df.columns:
        df['business_maturity'] = pd.cut(
            df['business_age_months'], 
            bins=[0, 12, 36, 60, np.inf],
            labels=['new', 'growing', 'established', 'mature']
        )
    
    # 5. Owner Age Category
    if 'owner_age' in df.columns:
        df['owner_age_category'] = pd.cut(
            df['owner_age'],
            bins=[0, 30, 45, 60, np.inf],
            labels=['young', 'middle', 'senior', 'elderly']
        )
    
    # Convert object columns to category
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    
    return df

@app.route('/')
def serve():
    """Serve the React app"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict financial health from input data
    
    Expected JSON input:
    {
        "country": "Zimbabwe",
        "owner_age": 35,
        "personal_income": 5000,
        "business_expenses": 3000,
        "business_age_months": 24,
        "has_bank_account": "Yes",
        "has_loan_account": "No",
        "has_internet_banking": "Yes",
        "has_debit_card": "Yes",
        "medical_insurance": "Have now",
        "funeral_insurance": "Never had",
        "compliance_income_tax": "Yes"
    }
    """
    try:
        if model_data is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please ensure the model file exists and is valid'
            }), 500
        
        # Get input data
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({
                'error': 'No input data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['country', 'owner_age', 'personal_income', 'business_expenses']
        missing_fields = [f for f in required_fields if f not in input_data or input_data[f] == '']
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Convert numeric fields
        numeric_fields = ['owner_age', 'personal_income', 'business_expenses', 'business_age_months']
        for field in numeric_fields:
            if field in input_data and input_data[field] != '':
                try:
                    input_data[field] = float(input_data[field])
                except ValueError:
                    return jsonify({
                        'error': f'Invalid numeric value for {field}'
                    }), 400
        
        # Create features
        feature_df = create_features(input_data)
        
        # Ensure all expected features are present
        expected_features = model_data['feature_columns']
        for feat in expected_features:
            if feat not in feature_df.columns:
                feature_df[feat] = np.nan
        
        # Select only the features used in training, in the correct order
        X = feature_df[expected_features]
        
        # Make prediction
        prediction = model_data['model'].predict(X)[0]
        prediction_proba = model_data['model'].predict_proba(X)[0]
        
        # Calculate additional metrics
        profit_margin = ((input_data.get('personal_income', 0) - input_data.get('business_expenses', 0)) / 
                        input_data.get('personal_income', 1)) if input_data.get('personal_income', 0) > 0 else 0
        
        # Map prediction to health level
        health_mapping = {
            0: 'Needs Improvement',
            1: 'Fair',
            2: 'Good',
            3: 'Excellent'
        }
        
        response = {
            'prediction': int(prediction),
            'health_level': health_mapping.get(int(prediction), 'Unknown'),
            'confidence': float(max(prediction_proba)),
            'probabilities': {
                'needs_improvement': float(prediction_proba[0]) if len(prediction_proba) > 0 else 0,
                'fair': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0,
                'good': float(prediction_proba[2]) if len(prediction_proba) > 2 else 0,
                'excellent': float(prediction_proba[3]) if len(prediction_proba) > 3 else 0
            },
            'metrics': {
                'profit_margin': round(profit_margin * 100, 2),
                'financial_access_score': float(feature_df['financial_access_score'].iloc[0]) if 'financial_access_score' in feature_df.columns else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple records
    
    Expected JSON input:
    {
        "records": [
            {...record1...},
            {...record2...}
        ]
    }
    """
    try:
        if model_data is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        input_data = request.get_json()
        records = input_data.get('records', [])
        
        if not records:
            return jsonify({
                'error': 'No records provided'
            }), 400
        
        predictions = []
        
        for idx, record in enumerate(records):
            try:
                feature_df = create_features(record)
                expected_features = model_data['feature_columns']
                
                for feat in expected_features:
                    if feat not in feature_df.columns:
                        feature_df[feat] = np.nan
                
                X = feature_df[expected_features]
                prediction = model_data['model'].predict(X)[0]
                
                predictions.append({
                    'record_index': idx,
                    'id': record.get('ID', f'record_{idx}'),
                    'prediction': int(prediction)
                })
            except Exception as e:
                predictions.append({
                    'record_index': idx,
                    'id': record.get('ID', f'record_{idx}'),
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': predictions,
            'total_records': len(records),
            'successful': len([p for p in predictions if 'error' not in p])
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        print("⚠️  Warning: Model not loaded. Please train the model first.")
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)