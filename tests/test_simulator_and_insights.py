"""
Unit tests for Scenario Simulator, AI Insights Discovery Engine, and Demographic Comparison.
"""

import pandas as pd
import numpy as np
import pytest
from modules.scenario_simulator import run_scenario_simulation
from modules.insights_engine import discover_automated_insights
from modules.demographic_comparison import compare_segmentation_approaches

@pytest.fixture
def sample_clustered_data():
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "customer_id": [f"CUST-{i:04d}" for i in range(n)],
        "segment_code": np.random.choice(["SEG-1", "SEG-2", "SEG-3"], n),
        "cluster_id": np.random.choice([0, 1, 2], n),
        "age": np.random.randint(18, 70, n),
        "gender": np.random.choice(["Female", "Male", "Non-Binary"], n),
        "location": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Miami"], n),
        "recency_days": np.random.uniform(5, 120, n),
        "purchase_frequency": np.random.randint(1, 12, n),
        "monetary_value": np.random.uniform(50, 3000, n),
        "avg_order_value": np.random.uniform(30, 250, n),
        "discount_usage_pct": np.random.uniform(0, 40, n),
        "total_interactions": np.random.randint(2, 60, n),
        "digital_activity_count": np.random.randint(1, 40, n),
        "campaign_response_rate": np.random.uniform(0, 0.4, n),
        "estimated_clv": np.random.uniform(200, 5000, n)
    })

def test_scenario_simulator(sample_clustered_data):
    res = run_scenario_simulation(
        sample_clustered_data,
        retention_boost_pct=15.0,
        repeat_rate_boost_pct=10.0,
        discount_reduction_pct=20.0,
        window_shopper_conversion_pct=5.0,
        marketing_budget=10000.0,
        gross_margin_pct=40.0
    )
    
    assert "baseline" in res
    assert "scenario" in res
    assert "total_revenue" in res["scenario"]
    assert "incremental_revenue" in res["scenario"]
    assert "incremental_gross_profit" in res["scenario"]
    assert "net_incremental_profit" in res["scenario"]
    assert "projected_roi_pct" in res["scenario"]
    assert res["scenario"]["total_revenue"] >= res["baseline"]["revenue"]

def test_demographic_comparison(sample_clustered_data):
    res = compare_segmentation_approaches(sample_clustered_data)
    
    assert "Demographic (Age Cohorts)" in res
    assert "Demographic (Geographic Regions)" in res
    assert "Behavioral (Heuristic RFM Tiers)" in res
    assert "Behavioral (ML Clustering Engine)" in res
    
    ml_res = res["Behavioral (ML Clustering Engine)"]
    assert "summary_table" in ml_res
    assert "f_statistic_monetary" in ml_res
    assert isinstance(ml_res["summary_table"], pd.DataFrame)

def test_insights_engine(sample_clustered_data):
    cluster_summary = pd.DataFrame([
        {"segment_code": "SEG-1", "customer_count": 50, "customer_pct": 33.3, "total_revenue": 50000, "revenue_share_pct": 50.0, "avg_recency": 20, "avg_frequency": 6, "avg_monetary": 1000, "avg_discount_pct": 5},
        {"segment_code": "SEG-2", "customer_count": 50, "customer_pct": 33.3, "total_revenue": 35000, "revenue_share_pct": 35.0, "avg_recency": 60, "avg_frequency": 4, "avg_monetary": 700, "avg_discount_pct": 15},
        {"segment_code": "SEG-3", "customer_count": 50, "customer_pct": 33.3, "total_revenue": 15000, "revenue_share_pct": 15.0, "avg_recency": 90, "avg_frequency": 1, "avg_monetary": 300, "avg_discount_pct": 35}
    ])
    
    insights = discover_automated_insights(
        sample_clustered_data,
        pd.DataFrame(),
        cluster_summary
    )
    
    assert isinstance(insights, list)
    assert len(insights) >= 1
    assert "category" in insights[0]
    assert "title" in insights[0]
    assert "recommended_action" in insights[0]
    assert "interpretation" in insights[0]
