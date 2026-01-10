"""
Feature Engineering Utilities
Reusable feature engineering functions
"""

import pandas as pd
import numpy as np

def create_profit_margin(df):
    """Calculate profit margin ratio"""
    return np.where(
        (df['personal_income'].notna()) & 
        (df['business_expenses'].notna()) & 
        (df['personal_income'] != 0),
        (df['personal_income'] - df['business_expenses']) / df['personal_income'],
        np.nan
    )

def create_financial_access_score(df):
    """Calculate financial access score"""
    features = [ 'has_loan_account', 'has_internet_banking',
        'has_debit_card', 'medical_insurance', 'funeral_insurance'
    ]
    
    def calc_score(row):
        score = 0
        valid = 0
        for feat in features:
            if feat in row and pd.notna(row[feat]):
                valid += 1
                if row[feat] in ['Yes', 'Have now', 'have now']:
                    score += 1
                elif 'Used to have' in str(row[feat]):
                    score += 0.5
        return score / valid if valid > 0 else np.nan
    
    return df.apply(calc_score, axis=1)
