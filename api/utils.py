"""
Utility functions for the API
"""

import pandas as pd
import numpy as np

def validate_input(data, required_fields):
    """Validate input data"""
    missing = [f for f in required_fields if f not in data or data[f] == '']
    if missing:
        return False, missing
    return True, []

def convert_to_numeric(data, numeric_fields):
    """Convert specified fields to numeric"""
    converted = data.copy()
    for field in numeric_fields:
        if field in converted and converted[field] != '':
            try:
                converted[field] = float(converted[field])
            except ValueError:
                raise ValueError(f"Invalid numeric value for {field}")
    return converted
