"""
Unit tests for SQLite database persistence and report generation (Excel & PDF).
"""

import os
import tempfile
import pandas as pd
import pytest
from modules.database import (
    init_db,
    save_segmentation_run,
    list_saved_runs,
    load_customer_assignments,
    get_run_recommendations
)
from modules.reporting import generate_excel_report, generate_pdf_report

@pytest.fixture
def temp_db_path():
    temp_dir = tempfile.mkdtemp()
    db_file = os.path.join(temp_dir, "test_segmentiq.db")
    init_db(db_file)
    yield db_file
    if os.path.exists(db_file):
        os.remove(db_file)

@pytest.fixture
def mock_export_data():
    df_clustered = pd.DataFrame({
        "customer_id": [f"CUST-{i:03d}" for i in range(10)],
        "segment_code": ["SEG-1"] * 5 + ["SEG-2"] * 5,
        "cluster_id": [0] * 5 + [1] * 5,
        "recency_days": [15.0] * 5 + [45.0] * 5,
        "purchase_frequency": [8] * 5 + [2] * 5,
        "monetary_value": [1200.0] * 5 + [300.0] * 5,
        "avg_order_value": [150.0] * 5 + [150.0] * 5,
        "discount_usage_pct": [5.0] * 5 + [20.0] * 5,
        "total_interactions": [25] * 5 + [10] * 5,
        "estimated_clv": [2500.0] * 5 + [600.0] * 5,
        "pca_1": [1.5] * 5 + [-1.5] * 5,
        "pca_2": [0.2] * 5 + [-0.2] * 5
    })
    
    cluster_summary = pd.DataFrame([
        {"segment_code": "SEG-1", "customer_count": 5, "customer_pct": 50.0, "total_revenue": 6000.0, "revenue_share_pct": 80.0, "avg_recency": 15.0, "avg_frequency": 8.0, "avg_monetary": 1200.0, "avg_discount_pct": 5.0},
        {"segment_code": "SEG-2", "customer_count": 5, "customer_pct": 50.0, "total_revenue": 1500.0, "revenue_share_pct": 20.0, "avg_recency": 45.0, "avg_frequency": 2.0, "avg_monetary": 300.0, "avg_discount_pct": 20.0}
    ])
    
    personas = {
        "is_llm_generated": False,
        "personas": {
            "SEG-1": {
                "persona_title": "High-Value Champions",
                "short_description": "Affluent frequent buyers",
                "dominant_behaviors": ["High AOV", "Frequent orders"],
                "spending_profile": "Generates 80% of revenue",
                "engagement_pattern": "Active email subscriber",
                "hypothesized_needs": ["VIP access", "Fast concierge"],
                "retention_barriers": ["Stockouts"],
                "confidence_assessment": "High"
            }
        }
    }
    
    strategies = {
        "SEG-1": {
            "segment_title": "High-Value Champions",
            "lifecycle_stage": "VIP Retention",
            "campaign_strategy": {
                "objective": "Maximize LTV",
                "channel": "Concierge Email",
                "core_message": "VIP private access",
                "incentive": "Complimentary gift",
                "timing": "Tuesday morning"
            },
            "justification": {
                "behavioral_evidence": "Top 10% spenders",
                "marketing_objective": "Retention",
                "recommended_action": "White glove concierge",
                "expected_direction_of_impact": "+15% CLV",
                "kpi_to_monitor": "CLV",
                "validation_method": "Holdout test",
                "confidence_level": "High"
            }
        }
    }
    
    metrics = {
        "algorithm": "K-Means",
        "n_clusters": 2,
        "features_used": ["recency_days", "purchase_frequency", "monetary_value"],
        "silhouette_score": 0.45,
        "davies_bouldin_index": 0.85
    }
    
    kpis = {
        "total_customers": 10,
        "total_revenue": 7500.0,
        "avg_aov": 150.0
    }
    
    return df_clustered, cluster_summary, personas, strategies, metrics, kpis

def test_database_operations(temp_db_path, mock_export_data):
    df_clustered, cluster_summary, personas, strategies, metrics, kpis = mock_export_data
    run_id = "run_test_001"
    
    success = save_segmentation_run(
        run_id=run_id,
        df_clustered=df_clustered,
        metrics=metrics,
        strategies=strategies,
        db_path=temp_db_path
    )
    assert success is True
    
    runs = list_saved_runs(db_path=temp_db_path)
    assert not runs.empty
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == run_id
    
    assignments = load_customer_assignments(run_id, db_path=temp_db_path)
    assert len(assignments) == 10
    assert "segment_code" in assignments.columns
    
    recs = get_run_recommendations(run_id, db_path=temp_db_path)
    assert len(recs) == 1
    assert recs.iloc[0]["segment_code"] == "SEG-1"

def test_excel_export(mock_export_data):
    df_clustered, cluster_summary, personas, strategies, metrics, kpis = mock_export_data
    excel_bytes = generate_excel_report(
        df_clustered, cluster_summary, personas, strategies, metrics
    )
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 1000

def test_pdf_export(mock_export_data):
    df_clustered, cluster_summary, personas, strategies, metrics, kpis = mock_export_data
    pdf_bytes = generate_pdf_report(
        df_clustered, cluster_summary, personas, strategies, metrics, kpis
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
