"""
SegmentIQ AI - SQLite Persistence Manager
Stores segmentation runs, customer cluster assignments, and marketing strategies.
"""

import sqlite3
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "segmentiq.db")

def init_db(db_path: str = DB_PATH):
    """Initializes SQLite schema for SegmentIQ AI."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS segmentation_runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT,
        algorithm TEXT,
        n_clusters INTEGER,
        features_used TEXT,
        silhouette_score REAL,
        davies_bouldin_index REAL,
        total_customers INTEGER,
        total_revenue REAL
    )
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS customer_segment_assignments (
        run_id TEXT,
        customer_id TEXT,
        segment_code TEXT,
        cluster_id INTEGER,
        recency_days REAL,
        purchase_frequency REAL,
        monetary_value REAL,
        avg_order_value REAL,
        pca_1 REAL,
        pca_2 REAL,
        PRIMARY KEY (run_id, customer_id)
    )
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS saved_recommendations (
        run_id TEXT,
        segment_code TEXT,
        segment_title TEXT,
        lifecycle_stage TEXT,
        campaign_objective TEXT,
        recommended_channel TEXT,
        recommended_offer TEXT,
        PRIMARY KEY (run_id, segment_code)
    )
    """)
    
    conn.commit()
    conn.close()

def save_segmentation_run(
    run_id: str,
    df_clustered: pd.DataFrame,
    metrics: Dict[str, Any],
    strategies: Dict[str, Any],
    db_path: str = DB_PATH
) -> bool:
    """Saves a complete segmentation run and customer assignments to SQLite."""
    try:
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # 1. Insert run metadata
        cur.execute("""
        INSERT OR REPLACE INTO segmentation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics.get("algorithm", "K-Means"),
            metrics.get("n_clusters", 0),
            json.dumps(metrics.get("features_used", [])),
            metrics.get("silhouette_score", 0.0),
            metrics.get("davies_bouldin_index", 0.0),
            len(df_clustered),
            float(df_clustered["monetary_value"].sum()) if "monetary_value" in df_clustered.columns else 0.0
        ))
        
        # 2. Insert customer assignments
        assign_rows = []
        for _, row in df_clustered.iterrows():
            assign_rows.append((
                run_id,
                str(row["customer_id"]),
                str(row.get("segment_code", "Unknown")),
                int(row.get("cluster_id", 0)),
                float(row.get("recency_days", 0.0)),
                float(row.get("purchase_frequency", 0.0)),
                float(row.get("monetary_value", 0.0)),
                float(row.get("avg_order_value", 0.0)),
                float(row.get("pca_1", 0.0)),
                float(row.get("pca_2", 0.0))
            ))
            
        cur.executemany("""
        INSERT OR REPLACE INTO customer_segment_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, assign_rows)
        
        # 3. Insert recommendations
        rec_rows = []
        for code, strat in strategies.items():
            camp = strat.get("campaign_strategy", {})
            rec_rows.append((
                run_id,
                code,
                strat.get("segment_title", ""),
                strat.get("lifecycle_stage", ""),
                camp.get("objective", ""),
                camp.get("channel", ""),
                camp.get("incentive", "")
            ))
            
        cur.executemany("""
        INSERT OR REPLACE INTO saved_recommendations VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rec_rows)
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving run to SQLite: {e}")
        return False

def list_saved_runs(db_path: str = DB_PATH) -> pd.DataFrame:
    """Returns list of previous runs from SQLite."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        df_runs = pd.read_sql("SELECT * FROM segmentation_runs ORDER BY created_at DESC", conn)
        conn.close()
        return df_runs
    except Exception:
        conn.close()
        return pd.DataFrame()

def load_customer_assignments(run_id: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Loads customer cluster assignments for a specific run ID."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT * FROM customer_segment_assignments WHERE run_id = ?"
        df_assignments = pd.read_sql(query, conn, params=(run_id,))
        conn.close()
        return df_assignments
    except Exception:
        conn.close()
        return pd.DataFrame()

def get_run_recommendations(run_id: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Loads saved marketing strategies for a specific run ID."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT * FROM saved_recommendations WHERE run_id = ?"
        df_recs = pd.read_sql(query, conn, params=(run_id,))
        conn.close()
        return df_recs
    except Exception:
        conn.close()
        return pd.DataFrame()

