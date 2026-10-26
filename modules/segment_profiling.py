"""
SegmentIQ AI - Segment Profiling & Comparison Module
Computes deep segment profiles, normalized radar scores, product category affinities,
and behavioral distributions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def compute_segment_radar_data(
    df_clustered: pd.DataFrame,
    features: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Min-max normalizes behavioral features across the population to 0-100 scale,
    then aggregates mean per segment for radar / spider charts.
    """
    if features is None:
        features = [
            "recency_days", "purchase_frequency", "monetary_value",
            "avg_order_value", "discount_usage_pct", "total_interactions",
            "digital_activity_count"
        ]
        
    available = [f for f in features if f in df_clustered.columns]
    if not available:
        return pd.DataFrame(), []
        
    # Invert recency so that higher value = fresher/better (Recency Score)
    df_scaled = df_clustered.copy()
    if "recency_days" in available:
        max_r = df_scaled["recency_days"].max()
        df_scaled["recency_score"] = max_r - df_scaled["recency_days"]
        radar_features = [f if f != "recency_days" else "recency_score" for f in available]
    else:
        radar_features = available
        
    # Scale each feature 0-100
    for col in radar_features:
        col_min = df_scaled[col].min()
        col_max = df_scaled[col].max()
        if col_max > col_min:
            df_scaled[col] = ((df_scaled[col] - col_min) / (col_max - col_min)) * 100
        else:
            df_scaled[col] = 50.0
            
    radar_df = df_scaled.groupby("segment_code")[radar_features].mean().round(1).reset_index()
    return radar_df, radar_features

def compute_segment_category_affinity(
    df_clustered: pd.DataFrame,
    df_transactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins clustered customer assignments with transaction records to calculate
    category spend share and transaction share by segment.
    """
    if df_transactions is None or df_transactions.empty:
        return pd.DataFrame()
        
    merged = df_transactions.merge(
        df_clustered[["customer_id", "segment_code"]],
        on="customer_id",
        how="inner"
    )
    
    if merged.empty or "product_category" not in merged.columns:
        return pd.DataFrame()
        
    cat_summary = (
        merged.groupby(["segment_code", "product_category"])
        .agg(
            order_count=("transaction_id", "count"),
            total_spend=("transaction_amount", "sum")
        )
        .reset_index()
    )
    
    # Calculate share within each segment
    seg_spend = cat_summary.groupby("segment_code")["total_spend"].transform("sum")
    cat_summary["spend_share_pct"] = round((cat_summary["total_spend"] / seg_spend) * 100, 2)
    
    return cat_summary

def compute_demographic_distributions(df_clustered: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculates distribution of demographic variables across segments to explore
    if segments have unique demographic leanings or if demographics are broad.
    """
    results = {}
    demo_cols = ["gender", "location", "acquisition_channel", "customer_type"]
    
    for col in demo_cols:
        if col in df_clustered.columns:
            ct = pd.crosstab(
                df_clustered["segment_code"],
                df_clustered[col],
                normalize="index"
            ) * 100
            results[col] = ct.round(1)
            
    return results

