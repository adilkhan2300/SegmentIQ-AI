"""
Tests for LLM and Rule-Based Personas and Marketing Strategies
"""

import pandas as pd
from modules.llm_personas import generate_personas
from modules.marketing_recommendations import generate_marketing_strategies

def test_rule_based_personas_without_api_key():
    cluster_summary = pd.DataFrame([
        {
            "segment_code": "Segment 0",
            "customer_count": 150,
            "pct_of_customers": 30.0,
            "revenue_share_pct": 55.0,
            "recency_days": 18.0,
            "purchase_frequency": 9.5,
            "monetary_value": 1850.0,
            "avg_order_value": 194.7,
            "discount_usage_pct": 3.2,
            "total_interactions": 28.0
        },
        {
            "segment_code": "Segment 1",
            "customer_count": 200,
            "pct_of_customers": 40.0,
            "revenue_share_pct": 20.0,
            "recency_days": 125.0,
            "purchase_frequency": 2.1,
            "monetary_value": 180.0,
            "avg_order_value": 85.7,
            "discount_usage_pct": 25.0,
            "total_interactions": 4.5
        }
    ])
    
    overall_stats = {
        "total_customers": 350,
        "avg_aov": 120.0,
        "avg_freq": 5.0,
        "avg_recency": 60.0
    }
    
    # Run with api_key=None -> should use Rule-Based Engine cleanly
    res = generate_personas(cluster_summary, overall_stats, api_key=None)
    assert res["generator_type"] == "Rule-Based Behavioral Persona Engine"
    assert "Segment 0" in res["personas"]
    assert "Segment 1" in res["personas"]
    
    p0 = res["personas"]["Segment 0"]
    assert "VIP" in p0["persona_title"] or "High-Value" in p0["persona_title"]
    assert len(p0["hypothesized_needs"]) > 0
    assert "Hypothesis:" in p0["hypothesized_needs"][0]
    
    # Test marketing recommendations
    strat_dict = generate_marketing_strategies(cluster_summary, res)
    assert "Segment 0" in strat_dict
    s0 = strat_dict["Segment 0"]
    assert "justification" in s0
    assert "validation_method" in s0["justification"]
    assert "expected_direction_of_impact" in s0["justification"]
