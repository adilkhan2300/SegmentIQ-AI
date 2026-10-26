"""
SegmentIQ AI - Demographic vs. Behavioral Segmentation Comparison Module
Provides side-by-side empirical comparison between demographic cohorts (Age, Location)
and behavioral segments (RFM, ML Clusters) with ANOVA variance decomposition.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple

def compare_segmentation_approaches(df_clustered: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comparative groupings:
    1. Demographic: Age Tiers (<25, 25-34, 35-49, 50+)
    2. Demographic: Location Regions
    3. Behavioral: RFM Heuristic Tiers
    4. Behavioral: ML Clusters
    Calculates metrics and ANOVA F-statistic on Monetary Value and Frequency.
    """
    df = df_clustered.copy()
    
    # 1. Create Age Group
    if "age" in df.columns:
        bins = [0, 25, 35, 50, 120]
        labels = ["Gen Z (<25)", "Young Adult (25-34)", "Mid-Career (35-49)", "Mature (50+)"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    else:
        df["age_group"] = "Unknown"
        
    # 2. Location group (top 4 or region)
    if "location" in df.columns:
        # Simplify location to city or top 5
        top_locs = df["location"].value_counts().nlargest(4).index
        df["location_group"] = df["location"].apply(lambda x: x if x in top_locs else "Other Regions")
    else:
        df["location_group"] = "Unknown"
        
    # 3. Traditional RFM Score Quintiles
    # Recency score 1-4, Frequency 1-4, Monetary 1-4
    if {"recency_days", "purchase_frequency", "monetary_value"}.issubset(df.columns):
        # Recency (lower days = higher score)
        df["r_tier"] = pd.qcut(df["recency_days"], q=4, labels=[4, 3, 2, 1], duplicates="drop")
        df["f_tier"] = pd.qcut(df["purchase_frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4], duplicates="drop")
        df["m_tier"] = pd.qcut(df["monetary_value"].rank(method="first"), q=4, labels=[1, 2, 3, 4], duplicates="drop")
        
        # Heuristic RFM category
        def classify_rfm(row):
            r, f, m = int(row["r_tier"]), int(row["f_tier"]), int(row["m_tier"])
            if r >= 3 and f >= 3 and m >= 3:
                return "RFM Champions"
            elif r >= 3 and f <= 2:
                return "RFM Recent Newcomers"
            elif r <= 2 and f >= 3:
                return "RFM At-Risk Loyalists"
            elif r <= 2 and f <= 2:
                return "RFM Inactive / Dormant"
            else:
                return "RFM Potential Regulars"
                
        df["rfm_group"] = df.apply(classify_rfm, axis=1)
    else:
        df["rfm_group"] = "RFM General"
        
    if "repeat_customer_flag" not in df.columns:
        if "purchase_frequency" in df.columns:
            df["repeat_customer_flag"] = (df["purchase_frequency"] > 1).astype(int)
        else:
            df["repeat_customer_flag"] = 0
            
    # Aggregate summaries for each approach
    approaches = {
        "Demographic (Age Cohorts)": "age_group",
        "Demographic (Geographic Regions)": "location_group",
        "Behavioral (Heuristic RFM Tiers)": "rfm_group",
        "Behavioral (ML Clustering Engine)": "segment_code"
    }
    
    comparisons = {}
    tot_rev = df["monetary_value"].sum() if "monetary_value" in df.columns else 1.0
    
    for label, col in approaches.items():
        grouped = df.groupby(col, observed=False).agg(
            customer_count=("customer_id", "count"),
            avg_monetary=("monetary_value", "mean"),
            avg_frequency=("purchase_frequency", "mean"),
            avg_aov=("avg_order_value", "mean"),
            avg_recency=("recency_days", "mean"),
            repeat_rate=("repeat_customer_flag", "mean"),
            total_spend=("monetary_value", "sum")
        ).reset_index()
        
        grouped["pct_customers"] = np.round((grouped["customer_count"] / len(df)) * 100, 1)
        grouped["revenue_share_pct"] = np.round((grouped["total_spend"] / tot_rev) * 100, 1)
        grouped["avg_monetary"] = np.round(grouped["avg_monetary"], 2)
        grouped["avg_frequency"] = np.round(grouped["avg_frequency"], 1)
        grouped["avg_aov"] = np.round(grouped["avg_aov"], 2)
        grouped["avg_recency"] = np.round(grouped["avg_recency"], 1)
        grouped["repeat_rate"] = np.round(grouped["repeat_rate"] * 100, 1)
        
        # Calculate ANOVA F-statistic for monetary_value across groups
        group_arrays = [group["monetary_value"].values for _, group in df.groupby(col, observed=False) if len(group) > 1]
        if len(group_arrays) > 1:
            f_stat_mon, p_mon = stats.f_oneway(*group_arrays)
        else:
            f_stat_mon, p_mon = 0.0, 1.0
            
        comparisons[label] = {
            "summary_table": grouped,
            "f_statistic_monetary": round(float(f_stat_mon), 2),
            "p_value_monetary": round(float(p_mon), 5),
            "group_count": len(grouped)
        }
        
    return comparisons
