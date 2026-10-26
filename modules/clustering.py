"""
SegmentIQ AI - Machine Learning Clustering & Dimensionality Reduction Engine
Implements K-Means, Agglomerative Clustering, DBSCAN, PCA, and cluster validation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

DEFAULT_SEGMENTATION_FEATURES = [
    "recency_days",
    "purchase_frequency",
    "monetary_value",
    "avg_order_value",
    "discount_usage_pct",
    "category_diversity_entropy",
    "total_interactions",
    "digital_activity_count"
]

def evaluate_candidate_k(
    df_features: pd.DataFrame,
    features: List[str] = DEFAULT_SEGMENTATION_FEATURES,
    k_range: range = range(2, 9),
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes Inertia, Silhouette Score, Davies-Bouldin, and Calinski-Harabasz
    across a range of k values to power the Elbow & Silhouette comparison charts.
    """
    valid_features = [f for f in features if f in df_features.columns]
    if not valid_features or len(df_features) < 10:
        return pd.DataFrame()
        
    X = df_features[valid_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    for k in k_range:
        if k >= len(df_features):
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
        db = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else 999.0
        ch = calinski_harabasz_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
        
        results.append({
            "k": k,
            "Inertia": round(float(km.inertia_), 2),
            "Silhouette Score": round(float(sil), 4),
            "Davies-Bouldin Index": round(float(db), 4),
            "Calinski-Harabasz Index": round(float(ch), 1)
        })
        
    return pd.DataFrame(results)

def run_clustering(
    df_features: pd.DataFrame,
    algorithm: str = "K-Means",
    n_clusters: int = 5,
    features: List[str] = DEFAULT_SEGMENTATION_FEATURES,
    dbscan_eps: float = 1.2,
    dbscan_min_samples: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes customer clustering using specified algorithm, scales features,
    projects to 2D with PCA, calculates evaluation metrics and cluster summaries.
    """
    valid_features = [f for f in features if f in df_features.columns]
    if not valid_features:
        valid_features = [c for c in df_features.select_dtypes(include=[np.number]).columns if c not in ["customer_id", "age"]]
        
    X_raw = df_features[valid_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 1. Model execution
    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
        labels = model.fit_predict(X_scaled)
        inertia = float(model.inertia_)
    elif algorithm == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
        inertia = None
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = model.fit_predict(X_scaled)
        inertia = None
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    # Remap noise points if DBSCAN
    unique_labels = sorted(list(set(labels)))
    n_discovered_clusters = len([l for l in unique_labels if l != -1])
    
    # Format cluster labels (e.g. "Cluster 0", "Cluster 1", or "Noise" for -1)
    cluster_names = []
    for l in labels:
        if l == -1:
            cluster_names.append("Noise / Outliers")
        else:
            cluster_names.append(f"Segment {l}")
            
    df_clustered = df_features.copy()
    df_clustered["cluster_id"] = labels
    df_clustered["segment_code"] = cluster_names
    
    # 2. PCA Dimensionality Reduction for 2D visualization
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    df_clustered["pca_1"] = np.round(X_pca[:, 0], 3)
    df_clustered["pca_2"] = np.round(X_pca[:, 1], 3)
    
    variance_explained = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    
    # 3. Validation Metrics
    # Filter out noise points for silhouette calculation if DBSCAN
    non_noise_mask = (labels != -1)
    if non_noise_mask.sum() > 1 and len(set(labels[non_noise_mask])) > 1:
        sil = float(silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
        db = float(davies_bouldin_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
        ch = float(calinski_harabasz_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
    else:
        sil, db, ch = 0.0, 999.0, 0.0
        
    metrics = {
        "algorithm": algorithm,
        "n_clusters": n_discovered_clusters,
        "features_used": valid_features,
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(db, 4),
        "calinski_harabasz_index": round(ch, 1),
        "inertia": round(inertia, 2) if inertia is not None else None,
        "pca_variance_ratio": variance_explained,
        "pca_total_variance": sum(variance_explained),
        "noise_points_count": int((labels == -1).sum()) if algorithm == "DBSCAN" else 0
    }
    
    return df_clustered, metrics

def get_cluster_summary(df_clustered: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
    """
    Computes interpretable segment-level averages and distributions in raw original units.
    """
    if features is None:
        features = [
            "recency_days", "purchase_frequency", "monetary_value", "avg_order_value",
            "discount_usage_pct", "total_interactions", "digital_activity_count", "estimated_clv"
        ]
        
    available = [f for f in features if f in df_clustered.columns]
    
    summary = df_clustered.groupby("segment_code")[available].mean().round(2)
    summary["customer_count"] = df_clustered.groupby("segment_code").size()
    summary["pct_of_customers"] = round((summary["customer_count"] / len(df_clustered)) * 100, 1)
    
    if "monetary_value" in df_clustered.columns:
        tot_rev = df_clustered["monetary_value"].sum()
        seg_rev = df_clustered.groupby("segment_code")["monetary_value"].sum()
        summary["total_revenue"] = seg_rev.round(2)
        summary["revenue_share_pct"] = round((seg_rev / tot_rev) * 100, 1) if tot_rev > 0 else 0.0
        
    return summary.reset_index()
