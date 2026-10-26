"""
Tests for Feature Engineering and RFM Calculations
"""

import pandas as pd
from datetime import datetime
from modules.feature_engineering import engineer_customer_features

def test_rfm_calculation_and_leakage():
    df_c = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "age": [30, 40],
        "registration_date": ["2024-01-01", "2024-01-01"]
    })
    df_t = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "customer_id": ["C1", "C1", "C2", "C1"],
        "transaction_date": ["2024-03-01", "2024-03-15", "2024-02-10", "2024-05-01"], # T4 is future
        "product_category": ["Electronics", "Electronics", "Books", "Electronics"],
        "quantity": [1, 2, 1, 1],
        "unit_price": [100.0, 50.0, 30.0, 100.0],
        "discount": [10.0, 0.0, 0.0, 0.0],
        "transaction_amount": [90.0, 100.0, 30.0, 100.0],
        "order_status": ["Completed", "Completed", "Completed", "Completed"]
    })
    
    # Set cutoff date to 2024-04-01 (should exclude T4 from calculation)
    cutoff = datetime(2024, 4, 1)
    df_features, feat_dict = engineer_customer_features(df_c, df_t, as_of_date=cutoff)
    
    assert len(df_features) == 2
    row_c1 = df_features[df_features["customer_id"] == "C1"].iloc[0]
    
    # C1 should have 2 completed orders before cutoff (T1 & T2), NOT 3
    assert row_c1["purchase_frequency"] == 2
    assert row_c1["monetary_value"] == 190.0
    assert row_c1["avg_order_value"] == 95.0
    assert row_c1["repeat_customer_flag"] == 1
    assert row_c1["recency_days"] == (cutoff - datetime(2024, 3, 15)).days
    assert not feat_dict.empty

def test_refund_and_cancelled_handling():
    df_c = pd.DataFrame({"customer_id": ["C1"], "registration_date": ["2024-01-01"]})
    df_t = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "customer_id": ["C1", "C1", "C1"],
        "transaction_date": ["2024-02-01", "2024-02-10", "2024-02-15"],
        "product_category": ["Apparel", "Apparel", "Apparel"],
        "quantity": [1, 1, 1],
        "unit_price": [50.0, 50.0, 50.0],
        "discount": [0.0, 0.0, 0.0],
        "transaction_amount": [50.0, 50.0, 50.0],
        "order_status": ["Completed", "Cancelled", "Refunded"]
    })
    df_features, _ = engineer_customer_features(df_c, df_t)
    row = df_features.iloc[0]
    # Completed orders should only count T1
    assert row["purchase_frequency"] == 1
    assert row["monetary_value"] == 50.0
    # Refund rate: 1 refunded out of 3 total attempted = 33.33%
    assert row["return_refund_rate"] == 33.33
