"""
Flask API for Financial Health Prediction
Serves trained model via REST endpoints
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__, static_folder='../static', static_url_path='')

# Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

MODEL_PATH = 'models/financial_health_model.pkl'
model_data = None

def load_model():
    """Load the trained model"""
    global model_data
    try:
        model_data = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def create_features(data):
    """Engineer features from input data"""
    df = pd.DataFrame([data])

    # Profit Margin
    df['profit_margin'] = np.where(
        (df['personal_income'].notna()) & 
        (df['business_expenses'].notna()) & 
        (df['personal_income'] != 0),
        (df['personal_income'] - df['business_expenses']) / df['personal_income'],
        np.nan
    )
    df['profit_margin'] = df['profit_margin'].clip(-1, 1)

    # Financial Access Score
    features = ['has_loan_account', 'has_internet_banking',
                'has_debit_card', 'medical_insurance', 'funeral_insurance']

    def calc_score(row):
        score = valid = 0
        for feat in features:
            val = row.get(feat)
            if pd.notna(val) and val != '':
                valid += 1
                if val in ['Yes', 'Have now', 'have now']:
                    score += 1
                elif 'Used to have' in str(val):
                    score += 0.5
        return score / valid if valid > 0 else np.nan

    df['financial_access_score'] = df.apply(calc_score, axis=1)

    # Income Expense Ratio
    df['income_expense_ratio'] = np.where(
        (df['business_expenses'].notna()) & (df['business_expenses'] > 0),
        df['personal_income'] / df['business_expenses'],
        np.nan
    )

    # Categories
    if 'business_age_months' in df.columns:
        df['business_maturity'] = pd.cut(
            df['business_age_months'], 
            bins=[0, 12, 36, 60, np.inf],
            labels=['new', 'growing', 'established', 'mature']
        )

    if 'owner_age' in df.columns:
        df['owner_age_category'] = pd.cut(
            df['owner_age'],
            bins=[0, 30, 45, 60, np.inf],
            labels=['young', 'middle', 'senior', 'elderly']
        )

    # Align categorical features with model training data
    if model_data and 'categorical_mappings' in model_data:
        for col, categories in model_data['categorical_mappings'].items():
            if col in df.columns:
                df[col] = pd.Categorical(df[col], categories=categories)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    return df

@app.route('/')
def serve():
    """Serve React app"""
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
    """Single prediction endpoint"""
    try:
        if model_data is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required = ['country', 'owner_age', 'personal_income', 'business_expenses']
        missing = [f for f in required if f not in data or data[f] == '']
        if missing:
            return jsonify({'error': 'Missing fields', 'missing': missing}), 400
        
        # Convert numerics
        for field in ['owner_age', 'personal_income', 'business_expenses', 'business_age_months']:
            if field in data and data[field] != '':
                data[field] = float(data[field])
        
        # Create features
        feature_df = create_features(data)
        
        # Ensure all features present
        for feat in model_data['feature_columns']:
            if feat not in feature_df.columns:
                feature_df[feat] = np.nan
        
        X = feature_df[model_data['feature_columns']]
        # print(f""Input features: {X.to_dict(orient='records')[0]}")
        print(f"Model: {model_data['model'].predict(X)[0]}")

        
        # Predict
        prediction = model_data['model'].predict(X)[0]
        print(f"Prediction: {prediction}")
        proba = model_data['model'].predict_proba(X)[0]

        
        health_levels = {0: 'Needs Improvement', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
        
        return jsonify({
            'prediction': int(prediction),
            'health_level': health_levels.get(int(prediction), 'Unknown'),
            'confidence': float(max(proba)),
            'probabilities': {
                'needs_improvement': float(proba[0]) if len(proba) > 0 else 0,
                'fair': float(proba[1]) if len(proba) > 1 else 0,
                'good': float(proba[2]) if len(proba) > 2 else 0,
                'excellent': float(proba[3]) if len(proba) > 3 else 0
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if model_data is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        records = data.get('records', [])
        
        predictions = []
        for idx, record in enumerate(records):
            try:
                feature_df = create_features(record)
                for feat in model_data['feature_columns']:
                    if feat not in feature_df.columns:
                        feature_df[feat] = np.nan
                
                X = feature_df[model_data['feature_columns']]
                pred = model_data['model'].predict(X)[0]
                
                predictions.append({
                    'index': idx,
                    'id': record.get('ID', f'record_{idx}'),
                    'prediction': int(pred)
                })
            except Exception as e:
                predictions.append({
                    'index': idx,
                    'id': record.get('ID', f'record_{idx}'),
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': predictions,
            'total': len(records),
            'successful': len([p for p in predictions if 'error' not in p])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    isModelLoaded = load_model()
    print(f"Model loaded: {isModelLoaded}")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
