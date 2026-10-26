"""
SegmentIQ AI - Customer Value and Retention Intelligence
Implements transparent business-rule risk scoring and predictive machine learning models
for churn risk and repeat purchase propensity with strict validation and error handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_rule_based_retention_risk(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns transparent business-rule based retention risk scores (0-100) and risk tiers:
    - High Risk (Dormant / Lapsed)
    - Medium Risk (Slowing Down)
    - Low Risk (Active & Healthy)
    """
    df = df_features.copy()
    
    # Calculate Risk Score components
    # 1. Recency Penalty (up to 40 pts): > 90 days is full risk
    rec_risk = np.clip(df["recency_days"] / 90.0, 0.0, 1.0) * 40.0
    
    # 2. Inactivity vs Historical Cadence Penalty (up to 25 pts)
    # If recency is more than 2x the average interval between purchases
    interval = df["avg_days_between_purchases"].replace(0, 30)
    interval_ratio = np.clip((df["recency_days"] / interval) - 1.0, 0.0, 2.0) / 2.0
    interval_risk = interval_ratio * 25.0
    
    # 3. Digital Engagement Silence (up to 20 pts)
    if "days_since_last_interaction" in df.columns:
        inter_risk = np.clip(df["days_since_last_interaction"] / 60.0, 0.0, 1.0) * 20.0
    else:
        inter_risk = 10.0
        
    # 4. Low Frequency / High Tenure Risk (up to 15 pts)
    # Long tenure but very few orders
    tenure_months = np.maximum(df["customer_tenure_days"] / 30.42, 1.0)
    velocity = df["purchase_frequency"] / tenure_months
    velocity_risk = np.where(velocity < 0.3, 15.0, np.where(velocity < 0.8, 8.0, 0.0))
    
    total_risk = np.round(rec_risk + interval_risk + inter_risk + velocity_risk, 1)
    df["retention_risk_score"] = np.clip(total_risk, 0.0, 100.0)
    
    # Assign Tiers
    conditions = [
        df["retention_risk_score"] >= 65.0,
        df["retention_risk_score"] >= 35.0
    ]
    choices = ["High Risk (Lapsed / At-Risk)", "Medium Risk (Slowing Cadence)", "Low Risk (Healthy / Active)"]
    df["retention_risk_tier"] = np.select(conditions, choices[:2], default=choices[2])
    
    return df

def train_retention_prediction_model(
    df_features: pd.DataFrame,
    target_type: str = "churn_risk",
    model_type: str = "Random Forest",
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Trains a classification model for retention risk or repeat purchase prediction.
    Includes feature importances, ROC-AUC, Precision, Recall, F1, and confusion matrix.
    """
    df = df_features.copy()
    
    # Define Target
    if target_type == "churn_risk":
        # Target: 1 if customer has not purchased in last 90 days or risk score > 60
        if "retention_risk_score" not in df.columns:
            df = compute_rule_based_retention_risk(df)
        y = (df["retention_risk_score"] >= 60.0).astype(int)
        target_name = "High Churn Risk (1 = At Risk, 0 = Active)"
    else:
        # Repeat purchase propensity (1 if frequency > 1)
        y = (df["purchase_frequency"] > 1).astype(int)
        target_name = "Repeat Customer Propensity (1 = Repeat, 0 = Single Order)"
        
    candidate_features = [
        "recency_days", "purchase_frequency", "avg_order_value", "discount_usage_pct",
        "customer_tenure_days", "total_interactions", "digital_activity_count",
        "campaign_response_rate", "category_diversity_entropy"
    ]
    features_used = [f for f in candidate_features if f in df.columns]
    
    # If target is derived directly from recency/frequency, drop the exact leakage variable
    if target_type == "churn_risk" and "recency_days" in features_used:
        # To avoid trivial identity leakage, drop recency_days from churn model
        features_used = [f for f in features_used if f != "recency_days"]
    elif target_type != "churn_risk" and "purchase_frequency" in features_used:
        features_used = [f for f in features_used if f != "purchase_frequency"]
        
    X = df[features_used].fillna(0).values
    
    if len(np.unique(y)) < 2:
        return {"error": "Target contains only one class. Cannot train classifier."}, pd.DataFrame()
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        importances = clf.feature_importances_
    else:
        clf = LogisticRegression(max_iter=500, random_state=random_state)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        importances = np.abs(clf.coef_[0])
        importances = importances / importances.sum()
        
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    feat_imp_df = pd.DataFrame({
        "Feature": features_used,
        "Importance": np.round(importances, 4)
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    # Store predictions on full dataset
    if model_type == "Random Forest":
        df["predicted_probability"] = np.round(clf.predict_proba(X)[:, 1], 3)
    else:
        df["predicted_probability"] = np.round(clf.predict_proba(scaler.transform(X))[:, 1], 3)
        
    metrics = {
        "target_name": target_name,
        "model_type": model_type,
        "roc_auc": round(float(auc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "test_sample_size": len(y_test),
        "features_used": features_used
    }
    
    return metrics, feat_imp_df, df
