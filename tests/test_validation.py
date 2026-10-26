"""
Tests for Data Validation Module
"""

import pandas as pd
import pytest
from modules.data_validation import validate_datasets, clean_and_prepare_data

def test_validation_clean_data():
    df_c = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "age": [25, 35, 45],
        "registration_date": ["2024-01-01", "2024-02-01", "2024-03-01"]
    })
    df_t = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "customer_id": ["C1", "C2", "C3"],
        "transaction_date": ["2024-04-01", "2024-04-05", "2024-04-10"],
        "transaction_amount": [100.0, 50.0, 200.0],
        "quantity": [1, 2, 1],
        "order_status": ["Completed", "Completed", "Refunded"]
    })
    
    result = validate_datasets(df_c, df_t)
    assert result["is_valid"] is True
    assert result["health_score"] >= 80

def test_validation_missing_columns():
    df_c = pd.DataFrame({"random_col": [1, 2, 3]})
    df_t = pd.DataFrame({"random_t": [4, 5, 6]})
    result = validate_datasets(df_c, df_t)
    assert result["is_valid"] is False
    assert len(result["critical_issues"]) > 0

def test_clean_and_prepare_data_deduplication():
    df_c = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "registration_date": ["2024-01-01", "2024-01-01", "2024-02-01"]
    })
    df_t = pd.DataFrame({
        "transaction_id": ["T1", "T1", "T2"],
        "customer_id": ["C1", "C1", "C2"],
        "transaction_date": ["2024-04-01", "2024-04-01", "2024-04-05"],
        "transaction_amount": [100.0, 100.0, 50.0],
        "order_status": ["completed", "completed", "completed"]
    })
    c_clean, t_clean, _ = clean_and_prepare_data(df_c, df_t)
    assert len(c_clean) == 2
    assert len(t_clean) == 2
    assert t_clean["order_status"].iloc[0] == "Completed"
