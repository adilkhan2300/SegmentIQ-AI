"""
SegmentIQ AI - Data Loader Module
Handles loading, caching, sample data provision, and CSV template generation.
"""

import os
import io
import pandas as pd
from typing import Dict, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_sample_datasets() -> Dict[str, pd.DataFrame]:
    """Loads default synthetic datasets if they exist, or triggers generation."""
    customers_path = os.path.join(DATA_DIR, "customers.csv")
    transactions_path = os.path.join(DATA_DIR, "transactions.csv")
    interactions_path = os.path.join(DATA_DIR, "interactions.csv")
    experiments_path = os.path.join(DATA_DIR, "campaign_experiments.csv")
    
    if not (os.path.exists(customers_path) and os.path.exists(transactions_path)):
        from modules.data_generator import generate_synthetic_data
        generate_synthetic_data(output_dir=DATA_DIR)
        
    df_customers = pd.read_csv(customers_path)
    df_transactions = pd.read_csv(transactions_path)
    df_interactions = pd.read_csv(interactions_path) if os.path.exists(interactions_path) else pd.DataFrame()
    df_experiments = pd.read_csv(experiments_path) if os.path.exists(experiments_path) else pd.DataFrame()
    
    return {
        "customers": df_customers,
        "transactions": df_transactions,
        "interactions": df_interactions,
        "experiments": df_experiments
    }

def read_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Reads uploaded CSV or Excel file safely."""
    if uploaded_file is None:
        return None
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif filename.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        else:
            return None
    except Exception as e:
        print(f"Error reading file {uploaded_file.name}: {e}")
        return None

def get_csv_template(schema_type: str) -> str:
    """Returns downloadable CSV template text with header and sample row."""
    if schema_type == "customers":
        return (
            "customer_id,age,gender,location,registration_date,acquisition_channel,customer_type\n"
            "CUST-0001,35,Female,\"New York, NY\",2024-06-15,Organic Search,Direct-to-Consumer\n"
            "CUST-0002,42,Male,\"Chicago, IL\",2024-07-20,Google Ads,Retail Member\n"
        )
    elif schema_type == "transactions":
        return (
            "transaction_id,customer_id,transaction_date,product_id,product_category,quantity,unit_price,discount,transaction_amount,payment_method,order_status\n"
            "TX-10001,CUST-0001,2024-08-10,PRD-ELE-105,Electronics,1,149.99,15.00,134.99,Credit Card,Completed\n"
            "TX-10002,CUST-0001,2024-09-12,PRD-APP-204,Apparel & Fashion,2,35.00,0.00,70.00,PayPal,Completed\n"
        )
    elif schema_type == "interactions":
        return (
            "interaction_id,customer_id,interaction_date,interaction_type,channel,campaign_id,engagement_value\n"
            "INT-50001,CUST-0001,2024-08-05,Email Open,Email,CAMP-FALL25,3.0\n"
            "INT-50002,CUST-0001,2024-08-08,Product View,Website,CAMP-FALL25,6.5\n"
        )
    elif schema_type == "campaign_experiments":
        return (
            "experiment_id,customer_id,campaign_name,variant,channel,converted,order_value,timestamp\n"
            "EXP-0001,CUST-0001,Q1 Retention Lift Test,Variant_B_Personalized,Email,1,120.50,2026-02-10\n"
        )
    return ""
