"""
Unit tests for retention risk scoring and machine learning churn/repeat models.
"""

import pandas as pd
import numpy as np
import pytest
from modules.retention_analysis import (
    compute_rule_based_retention_risk,
    train_retention_prediction_model
)

@pytest.fixture
def sample_feature_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "customer_id": [f"CUST-{i:04d}" for i in range(n)],
        "recency_days": np.random.uniform(5, 180, n),
        "purchase_frequency": np.random.randint(1, 15, n),
        "monetary_value": np.random.uniform(50, 2500, n),
        "avg_order_value": np.random.uniform(25, 200, n),
        "avg_days_between_purchases": np.random.uniform(10, 60, n),
        "customer_tenure_days": np.random.uniform(90, 720, n),
        "discount_usage_pct": np.random.uniform(0, 50, n),
        "total_interactions": np.random.randint(2, 50, n),
        "digital_activity_count": np.random.randint(1, 40, n),
        "campaign_response_rate": np.random.uniform(0, 0.4, n),
        "category_diversity_entropy": np.random.uniform(0, 2.0, n),
        "days_since_last_interaction": np.random.uniform(2, 90, n)
    })

def test_rule_based_retention_risk(sample_feature_data):
    df_risk = compute_rule_based_retention_risk(sample_feature_data)
    assert "retention_risk_score" in df_risk.columns
    assert "retention_risk_tier" in df_risk.columns
    assert df_risk["retention_risk_score"].min() >= 0.0
    assert df_risk["retention_risk_score"].max() <= 100.0
    assert set(df_risk["retention_risk_tier"].unique()).issubset({
        "High Risk (Lapsed / At-Risk)",
        "Medium Risk (Slowing Cadence)",
        "Low Risk (Healthy / Active)"
    })

def test_train_retention_churn_model(sample_feature_data):
    metrics, df_imp, df_pred = train_retention_prediction_model(
        sample_feature_data,
        target_type="churn_risk",
        model_type="Random Forest",
        random_state=42
    )
    assert "error" not in metrics
    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert isinstance(df_imp, pd.DataFrame)
    assert not df_imp.empty
    assert "Feature" in df_imp.columns
    assert "Importance" in df_imp.columns
    assert "predicted_probability" in df_pred.columns

def test_train_retention_repeat_model(sample_feature_data):
    metrics, df_imp, df_pred = train_retention_prediction_model(
        sample_feature_data,
        target_type="repeat_propensity",
        model_type="Logistic Regression",
        random_state=42
    )
    assert "error" not in metrics
    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert isinstance(df_imp, pd.DataFrame)
    assert not df_imp.empty
    assert "predicted_probability" in df_pred.columns
