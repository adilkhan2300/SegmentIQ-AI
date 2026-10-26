"""
Unit tests for campaign experimentation, A/B testing analysis, and power calculations.
"""

import pandas as pd
import numpy as np
import pytest
from modules.experiment_analysis import (
    analyze_ab_experiment,
    calculate_sample_size_requirement,
    simulate_hypothetical_campaign
)

@pytest.fixture
def sample_experiment_data():
    np.random.seed(42)
    n_a = 500
    n_b = 500
    
    ctrl = pd.DataFrame({
        "customer_id": [f"CUST-A-{i:04d}" for i in range(n_a)],
        "variant": "Variant_A_Control",
        "segment_code": np.random.choice(["SEG-1", "SEG-2", "SEG-3"], n_a),
        "converted": np.random.binomial(1, 0.08, n_a),
        "order_value": np.random.exponential(75, n_a)
    })
    
    treat = pd.DataFrame({
        "customer_id": [f"CUST-B-{i:04d}" for i in range(n_b)],
        "variant": "Variant_B_Personalized",
        "segment_code": np.random.choice(["SEG-1", "SEG-2", "SEG-3"], n_b),
        "converted": np.random.binomial(1, 0.13, n_b),
        "order_value": np.random.exponential(85, n_b)
    })
    
    return pd.concat([ctrl, treat], ignore_index=True)

def test_analyze_ab_experiment(sample_experiment_data):
    results = analyze_ab_experiment(
        sample_experiment_data,
        control_variant="Variant_A_Control",
        treatment_variant="Variant_B_Personalized"
    )
    
    assert "error" not in results
    assert "p_value" in results
    assert "relative_lift_pct" in results
    assert "is_statistically_significant" in results
    assert "ci_lower_pct" in results
    assert "ci_upper_pct" in results
    assert results["n_control"] == 500
    assert results["n_treatment"] == 500

def test_sample_size_requirement():
    res = calculate_sample_size_requirement(
        baseline_cr=0.08,
        mde_relative=0.20,
        power=0.80,
        significance=0.05
    )
    assert "required_sample_per_variant" in res
    assert res["required_sample_per_variant"] > 0
    assert res["total_sample_required"] == res["required_sample_per_variant"] * 2

def test_simulate_hypothetical_campaign():
    sim = simulate_hypothetical_campaign(
        target_segment_size=5000,
        channel="Email",
        budget=1000.0,
        baseline_conversion_rate=4.0,
        projected_lift_pct=25.0,
        average_order_value=90.0,
        gross_margin_pct=40.0
    )
    assert sim["target_audience_size"] == 5000
    assert sim["customers_reached"] > 0
    assert "projected_orders" in sim
    assert "incremental_orders" in sim
    assert "projected_revenue" in sim
    assert "net_profit" in sim
    assert "estimated_roi_pct" in sim
