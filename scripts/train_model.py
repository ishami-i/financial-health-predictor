"""
Financial Health Prediction - Model Training Script
Trains LightGBM model and ensures all test IDs are included in submission
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FinancialHealthPredictor:
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        self.model = None
        self.feature_columns = None
        
    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        self.train_df = pd.read_csv(f'{self.data_path}Train.csv')
        self.test_df = pd.read_csv(f'{self.data_path}Test.csv')
        self.sample_submission = pd.read_csv(f'{self.data_path}SampleSubmission.csv')
        print(f"✅ Train shape: {self.train_df.shape}")
        print(f"✅ Test shape: {self.test_df.shape}")
        print(f"✅ Sample submission shape: {self.sample_submission.shape}")
        
    def create_features(self, df):
        """Engineer features from raw data"""
        df = df.copy()
        
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
        available = [col for col in financial_features if col in df.columns]
        
        def calc_access_score(row):
            score = 0
            valid = 0
            for feat in available:
                val = row.get(feat)
                if pd.notna(val):
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
        
        return df
    
    def prepare_data(self):
        """Prepare training and test data"""
        print("\n🔧 Engineering features...")
        self.train_fe = self.create_features(self.train_df)
        self.test_fe = self.create_features(self.test_df)
        
        X = self.train_fe.drop(columns=['Target', 'ID'])
        y = self.train_fe['Target']
        X_test = self.test_fe.drop(columns=['ID'])
        
        self.feature_columns = X.columns.tolist()
        
        # Train-validation split FIRST
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=self.train_fe[['country', 'Target']]
        )
        
        # Now convert to category AFTER split, with identical categories
        categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
        
        for col in categorical_cols:
            # Get all unique categories from both train and val combined
            all_categories = list(set(X_train[col].unique()) | set(X_val[col].unique()))
            all_categories = sorted([c for c in all_categories if pd.notna(c)])
            
            # Apply the SAME categories to both train and val
            X_train[col] = pd.Categorical(X_train[col], categories=all_categories)
            X_val[col] = pd.Categorical(X_val[col], categories=all_categories)
            
            # Also apply to test set
            if col in X_test.columns:
                X_test[col] = pd.Categorical(X_test[col], categories=all_categories)
        
        return X_train, X_val, y_train, y_val, X_test, y
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the LightGBM model"""
        print("\n🚀 Training LightGBM model...")
        
        self.model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        
        self.model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
)

        
        y_pred = self.model.predict(X_val)
        # Ensure numpy arrays for compatibility with sklearn metrics
        y_pred = np.asarray(y_pred)
        y_val_arr = np.asarray(y_val)
        print("\n📊 Validation Results:")
        print(classification_report(y_val_arr, y_pred))
        print(f"\n✅ Weighted F1 Score: {f1_score(y_val_arr, y_pred, average='weighted'):.4f}")
        
    def predict_and_save(self, X_test):
        """Generate predictions with all test IDs"""
        print("\n🔮 Generating predictions...")
        # Ensure we have a model object; try loading a saved model if not present
        if self.model is None:
            model_path = os.path.join('models', 'financial_health_model.pkl')
            if os.path.exists(model_path):
                print("No in-memory model found, loading saved model from models/financial_health_model.pkl")
                model_data = joblib.load(model_path)
                self.model = model_data.get('model')
                saved_features = model_data.get('feature_columns')
                if self.model is None:
                    raise RuntimeError("Loaded file does not contain a valid model object.")
                if getattr(self, 'feature_columns', None) is None and saved_features is not None:
                    self.feature_columns = saved_features
            else:
                raise RuntimeError("Model is not trained and no saved model found. Call train() first or provide a saved model at models/financial_health_model.pkl")
        
        # Align X_test to the trained feature columns if available
        if getattr(self, 'feature_columns', None):
            # Add any missing columns as NaN so model.predict won't fail
            missing_cols = [c for c in self.feature_columns if c not in X_test.columns]
            for c in missing_cols:
                X_test[c] = np.nan
            # Keep only the features the model expects and in the same order
            X_test = X_test.reindex(columns=self.feature_columns)
        
        # Preserve categorical dtypes where possible (if they exist in the training data)
        if hasattr(self, 'train_fe'):
            for col in X_test.columns:
                if col in self.train_fe.columns and self.train_fe[col].dtype.name == 'category':
                    X_test[col] = X_test[col].astype('category')
        
        predictions = self.model.predict(X_test)
        predictions = np.asarray(predictions)
        
        # Create submission with ALL test IDs
        submission = pd.DataFrame({
            'ID': self.test_df['ID'].values,
            'Target': predictions
        })
        
        # Verify against sample submission
        missing_ids = set(self.sample_submission['ID']) - set(submission['ID'])
        if missing_ids:
            print(f"⚠️  WARNING: {len(missing_ids)} missing IDs detected!")
            default_pred = self.train_df['Target'].mode()[0]
            for mid in missing_ids:
                submission = pd.concat([
                    submission,
                    pd.DataFrame({'ID': [mid], 'Target': [default_pred]})
                ], ignore_index=True)
        
        # Ensure exact match with sample submission
        submission = submission.merge(
            self.sample_submission[['ID']], 
            on='ID', 
            how='right'
        )
        
        if submission['Target'].isna().any():
            default_pred = self.train_df['Target'].mode()[0]
            submission['Target'].fillna(default_pred, inplace=True)
        
        # Verify
        assert len(submission) == len(self.sample_submission), \
            f"Length mismatch: {len(submission)} vs {len(self.sample_submission)}"
        
        # Save
        submission.to_csv('submission.csv', index=False)
        print(f"\n✅ Submission saved: {len(submission)} predictions")
        print(f"Distribution:\n{submission['Target'].value_counts().sort_index()}")
        
        return submission
    
    def save_model(self):
        """Save model for production use"""
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, 'models/financial_health_model.pkl')
        print("\n✅ Model saved to models/financial_health_model.pkl")

def main():
    predictor = FinancialHealthPredictor(data_path='./data/')
    predictor.load_data()
    X_train, X_val, y_train, y_val, X_test, y = predictor.prepare_data()
    predictor.train(X_train, y_train, X_val, y_val)
    predictor.predict_and_save(X_test)
    predictor.save_model()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Submit 'submission.csv' to Zindi")
    print("2. Run 'python api/app.py' to start the web server")

if __name__ == "__main__":
    main()
