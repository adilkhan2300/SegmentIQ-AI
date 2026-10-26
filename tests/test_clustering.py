"""
Tests for ML Clustering Engine
"""

import pandas as pd
import numpy as np
from modules.clustering import run_clustering, evaluate_candidate_k, get_cluster_summary

def test_kmeans_clustering():
    np.random.seed(42)
    # Create 50 synthetic customer feature rows
    df_feat = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(50)],
        "recency_days": np.random.uniform(5, 120, 50),
        "purchase_frequency": np.random.uniform(1, 15, 50),
        "monetary_value": np.random.uniform(50, 1500, 50),
        "avg_order_value": np.random.uniform(20, 200, 50),
        "discount_usage_pct": np.random.uniform(0, 30, 50),
        "total_interactions": np.random.uniform(1, 40, 50)
    })
    
    df_clustered, metrics = run_clustering(df_feat, algorithm="K-Means", n_clusters=3)
    assert len(df_clustered) == 50
    assert "cluster_id" in df_clustered.columns
    assert "pca_1" in df_clustered.columns
    assert metrics["silhouette_score"] is not None
    assert metrics["n_clusters"] == 3
    
    summary = get_cluster_summary(df_clustered)
    assert len(summary) == 3
    assert "revenue_share_pct" in summary.columns

def test_agglomerative_and_dbscan():
    np.random.seed(42)
    df_feat = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(40)],
        "recency_days": np.random.uniform(5, 120, 40),
        "purchase_frequency": np.random.uniform(1, 15, 40),
        "monetary_value": np.random.uniform(50, 1500, 40)
    })
    
    # Agglomerative
    df_agg, m_agg = run_clustering(df_feat, algorithm="Agglomerative Clustering", n_clusters=3)
    assert m_agg["n_clusters"] == 3
    
    # DBSCAN
    df_db, m_db = run_clustering(df_feat, algorithm="DBSCAN", dbscan_eps=1.5, dbscan_min_samples=3)
    assert "cluster_id" in df_db.columns
    assert "noise_points_count" in m_db

def test_candidate_k_evaluation():
    np.random.seed(42)
    df_feat = pd.DataFrame({
        "recency_days": np.random.uniform(5, 120, 30),
        "purchase_frequency": np.random.uniform(1, 15, 30),
        "monetary_value": np.random.uniform(50, 1500, 30)
    })
    eval_df = evaluate_candidate_k(df_feat, k_range=range(2, 5))
    assert len(eval_df) == 3
    assert "Silhouette Score" in eval_df.columns
    assert "Inertia" in eval_df.columns
