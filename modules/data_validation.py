"""
SegmentIQ AI - Data Validation Module
Comprehensive quality checks, data cleaning, and consistency verification.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def validate_datasets(
    df_customers: pd.DataFrame,
    df_transactions: pd.DataFrame,
    df_interactions: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Performs full validation across customer, transaction, and interaction datasets.
    Returns audit results, issue counts, cleanliness scores, and recommendations.
    """
    issues = []
    warnings = []
    stats = {}
    
    # 1. Customers Validation
    c_issues = []
    if df_customers is None or df_customers.empty:
        c_issues.append("Customer dataset is empty or not provided.")
    else:
        req_cust_cols = ["customer_id"]
        for col in req_cust_cols:
            if col not in df_customers.columns:
                c_issues.append(f"Missing mandatory customer column: '{col}'.")
                
        # Duplicates
        cust_dup = df_customers["customer_id"].duplicated().sum() if "customer_id" in df_customers.columns else 0
        if cust_dup > 0:
            c_issues.append(f"Found {cust_dup} duplicate customer IDs.")
            
        # Missing values
        cust_nulls = df_customers.isnull().sum().to_dict()
        stats["customers_total"] = len(df_customers)
        stats["customers_duplicates"] = int(cust_dup)
        stats["customers_nulls"] = cust_nulls

    # 2. Transactions Validation
    t_issues = []
    if df_transactions is None or df_transactions.empty:
        t_issues.append("Transaction dataset is empty or not provided.")
    else:
        req_tx_cols = ["transaction_id", "customer_id", "transaction_date", "transaction_amount"]
        for col in req_tx_cols:
            if col not in df_transactions.columns:
                t_issues.append(f"Missing mandatory transaction column: '{col}'.")
                
        # Duplicate transaction IDs
        if "transaction_id" in df_transactions.columns:
            tx_dup = df_transactions["transaction_id"].duplicated().sum()
            if tx_dup > 0:
                t_issues.append(f"Found {tx_dup} duplicate transaction IDs.")
        else:
            tx_dup = 0
            
        # Negative / Invalid quantities or prices
        neg_amounts = 0
        if "transaction_amount" in df_transactions.columns:
            neg_amounts = (df_transactions["transaction_amount"] < 0).sum()
            if neg_amounts > 0:
                warnings.append(f"{neg_amounts} transactions have negative amounts (may indicate manual returns).")
                
        neg_qty = 0
        if "quantity" in df_transactions.columns:
            neg_qty = (df_transactions["quantity"] <= 0).sum()
            if neg_qty > 0:
                t_issues.append(f"{neg_qty} transactions have zero or negative quantity.")
                
        # Date parsing check
        invalid_dates = 0
        if "transaction_date" in df_transactions.columns:
            parsed_dates = pd.to_datetime(df_transactions["transaction_date"], errors="coerce")
            invalid_dates = parsed_dates.isnull().sum()
            if invalid_dates > 0:
                t_issues.append(f"{invalid_dates} invalid transaction dates detected.")
                
        # Status distribution
        status_dist = {}
        if "order_status" in df_transactions.columns:
            status_dist = df_transactions["order_status"].value_counts().to_dict()
        else:
            warnings.append("'order_status' column missing; assuming all records are Completed purchases.")
            
        # Foreign Key check: transactions customer_id in df_customers
        orphaned_tx_cust = 0
        if df_customers is not None and "customer_id" in df_customers.columns and "customer_id" in df_transactions.columns:
            cust_id_set = set(df_customers["customer_id"].dropna().unique())
            orphaned = df_transactions[~df_transactions["customer_id"].isin(cust_id_set)]
            orphaned_tx_cust = len(orphaned)
            if orphaned_tx_cust > 0:
                warnings.append(f"{orphaned_tx_cust} transactions belong to customer IDs not found in customers table.")
                
        stats["transactions_total"] = len(df_transactions)
        stats["transactions_duplicates"] = int(tx_dup)
        stats["transactions_invalid_dates"] = int(invalid_dates)
        stats["transactions_status_dist"] = status_dist
        stats["transactions_orphaned"] = int(orphaned_tx_cust)

    # 3. Interactions Validation (optional)
    i_issues = []
    if df_interactions is not None and not df_interactions.empty:
        req_int_cols = ["customer_id", "interaction_date"]
        for col in req_int_cols:
            if col not in df_interactions.columns:
                i_issues.append(f"Missing interaction column: '{col}'.")
                
        orphaned_int_cust = 0
        if df_customers is not None and "customer_id" in df_customers.columns and "customer_id" in df_interactions.columns:
            cust_id_set = set(df_customers["customer_id"].dropna().unique())
            orphaned = df_interactions[~df_interactions["customer_id"].isin(cust_id_set)]
            orphaned_int_cust = len(orphaned)
            if orphaned_int_cust > 0:
                warnings.append(f"{orphaned_int_cust} interactions belong to customer IDs not in customers table.")
                
        stats["interactions_total"] = len(df_interactions)
        stats["interactions_orphaned"] = int(orphaned_int_cust)
        if "interaction_type" in df_interactions.columns:
            stats["interaction_types"] = df_interactions["interaction_type"].value_counts().to_dict()
    else:
        stats["interactions_total"] = 0
        warnings.append("Interactions dataset not loaded. Engagement features will default to zero.")

    # Calculate overall Health Score (0 - 100)
    score = 100
    score -= min(30, len(c_issues) * 15)
    score -= min(40, len(t_issues) * 10)
    score -= min(20, len(warnings) * 4)
    health_score = max(10, score)
    
    return {
        "is_valid": len(c_issues) == 0 and len(t_issues) == 0,
        "health_score": health_score,
        "critical_issues": c_issues + t_issues + i_issues,
        "warnings": warnings,
        "stats": stats
    }

def clean_and_prepare_data(
    df_customers: pd.DataFrame,
    df_transactions: pd.DataFrame,
    df_interactions: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans datasets:
    - Parses dates safely
    - Drops duplicate IDs
    - Normalizes strings
    - Ensures consistent customer ID types
    """
    c_clean = df_customers.copy()
    t_clean = df_transactions.copy()
    i_clean = df_interactions.copy() if df_interactions is not None and not df_interactions.empty else pd.DataFrame()
    
    # 1. Customers clean
    if "customer_id" in c_clean.columns:
        c_clean["customer_id"] = c_clean["customer_id"].astype(str).str.strip()
        c_clean = c_clean.drop_duplicates(subset=["customer_id"])
    if "registration_date" in c_clean.columns:
        c_clean["registration_date"] = pd.to_datetime(c_clean["registration_date"], errors="coerce")
        
    # 2. Transactions clean
    if "customer_id" in t_clean.columns:
        t_clean["customer_id"] = t_clean["customer_id"].astype(str).str.strip()
    if "transaction_id" in t_clean.columns:
        t_clean["transaction_id"] = t_clean["transaction_id"].astype(str).str.strip()
        t_clean = t_clean.drop_duplicates(subset=["transaction_id"])
    if "transaction_date" in t_clean.columns:
        t_clean["transaction_date"] = pd.to_datetime(t_clean["transaction_date"], errors="coerce")
    if "transaction_amount" in t_clean.columns:
        t_clean["transaction_amount"] = pd.to_numeric(t_clean["transaction_amount"], errors="coerce").fillna(0.0)
    if "quantity" in t_clean.columns:
        t_clean["quantity"] = pd.to_numeric(t_clean["quantity"], errors="coerce").fillna(1)
    if "discount" in t_clean.columns:
        t_clean["discount"] = pd.to_numeric(t_clean["discount"], errors="coerce").fillna(0.0)
    if "order_status" in t_clean.columns:
        t_clean["order_status"] = t_clean["order_status"].astype(str).str.title().str.strip()
    else:
        t_clean["order_status"] = "Completed"
        
    # 3. Interactions clean
    if not i_clean.empty:
        if "customer_id" in i_clean.columns:
            i_clean["customer_id"] = i_clean["customer_id"].astype(str).str.strip()
        if "interaction_date" in i_clean.columns:
            i_clean["interaction_date"] = pd.to_datetime(i_clean["interaction_date"], errors="coerce")
        if "engagement_value" in i_clean.columns:
            i_clean["engagement_value"] = pd.to_numeric(i_clean["engagement_value"], errors="coerce").fillna(1.0)
            
    return c_clean, t_clean, i_clean
