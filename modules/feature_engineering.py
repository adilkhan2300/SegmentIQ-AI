"""
SegmentIQ AI - Behavioral Feature Engineering Pipeline
Converts raw transactions and interactions into customer-level behavioral and RFM metrics.
Includes data leakage prevention and complete feature dictionary metadata.
"""

import math
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

def calculate_shannon_entropy(series: pd.Series) -> float:
    """Calculates Shannon entropy for category diversity."""
    if series.empty:
        return 0.0
    counts = series.value_counts()
    probs = counts / counts.sum()
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return round(entropy, 3)

def engineer_customer_features(
    df_customers: pd.DataFrame,
    df_transactions: pd.DataFrame,
    df_interactions: Optional[pd.DataFrame] = None,
    as_of_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds customer-level behavioral feature matrix.
    Excludes future transactions/interactions to prevent target leakage.
    """
    c_df = df_customers.copy()
    t_df = df_transactions.copy()
    i_df = df_interactions.copy() if df_interactions is not None and not df_interactions.empty else pd.DataFrame()
    
    # 1. Establish Observation Cutoff Date
    if as_of_date is None:
        t_max = pd.to_datetime(t_df["transaction_date"]).max() if not t_df.empty else datetime.now()
        i_max = pd.to_datetime(i_df["interaction_date"]).max() if not i_df.empty else datetime.now()
        as_of_date = max(t_max, i_max)
    else:
        as_of_date = pd.to_datetime(as_of_date)
        
    # Filter out future records to prevent data leakage
    if not t_df.empty and "transaction_date" in t_df.columns:
        t_df["transaction_date"] = pd.to_datetime(t_df["transaction_date"])
        t_df = t_df[t_df["transaction_date"] <= as_of_date]
        
    if not i_df.empty and "interaction_date" in i_df.columns:
        i_df["interaction_date"] = pd.to_datetime(i_df["interaction_date"])
        i_df = i_df[i_df["interaction_date"] <= as_of_date]

    # Pre-split transactions by order status
    # Note: Completed purchases drive RFM. Refunded orders contribute to refund rate. Cancelled orders excluded.
    t_completed = t_df[t_df["order_status"] == "Completed"].copy()
    t_refunded = t_df[t_df["order_status"] == "Refunded"].copy()
    
    total_system_revenue = t_completed["transaction_amount"].sum() if not t_completed.empty else 1.0

    # 2. Iterate or Aggregate Customers
    customer_features = []
    cust_ids = c_df["customer_id"].unique()
    
    # Group transactions for fast lookup
    t_comp_by_cust = dict(tuple(t_completed.groupby("customer_id"))) if not t_completed.empty else {}
    t_ref_by_cust = dict(tuple(t_refunded.groupby("customer_id"))) if not t_refunded.empty else {}
    t_all_by_cust = dict(tuple(t_df.groupby("customer_id"))) if not t_df.empty else {}
    i_by_cust = dict(tuple(i_df.groupby("customer_id"))) if not i_df.empty else {}

    for idx, c_row in c_df.iterrows():
        cid = c_row["customer_id"]
        reg_date = pd.to_datetime(c_row.get("registration_date"))
        
        # Tenure
        if pd.notnull(reg_date):
            tenure_days = max(1, (as_of_date - reg_date).days)
        else:
            tenure_days = 90  # fallback assumption
            
        # Completed Transactions
        cust_t_comp = t_comp_by_cust.get(cid, pd.DataFrame())
        cust_t_all = t_all_by_cust.get(cid, pd.DataFrame())
        cust_t_ref = t_ref_by_cust.get(cid, pd.DataFrame())
        
        # RFM Metrics
        if not cust_t_comp.empty:
            last_tx_date = cust_t_comp["transaction_date"].max()
            recency = max(0, (as_of_date - last_tx_date).days)
            frequency = len(cust_t_comp)
            monetary = float(cust_t_comp["transaction_amount"].sum())
            aov = round(monetary / frequency, 2)
            total_items = cust_t_comp["quantity"].sum()
            avg_items_per_order = round(total_items / frequency, 2)
            
            # Discounts
            gross_spent = (cust_t_comp["unit_price"] * cust_t_comp["quantity"]).sum()
            total_discount = cust_t_comp["discount"].sum()
            discount_usage_pct = round((total_discount / gross_spent) * 100, 2) if gross_spent > 0 else 0.0
            
            # Category diversity & preferred category
            cats = cust_t_comp["product_category"]
            cat_entropy = calculate_shannon_entropy(cats)
            cat_unique_count = cats.nunique()
            preferred_cat = cats.mode().iloc[0] if not cats.empty else "None"
            
            # Purchase intervals
            if frequency > 1:
                sorted_dates = cust_t_comp["transaction_date"].sort_values()
                intervals = sorted_dates.diff().dt.days.dropna()
                avg_time_between_purchases = round(intervals.mean(), 1) if not intervals.empty else 0.0
            else:
                avg_time_between_purchases = tenure_days  # Single purchase: interval is tenure
        else:
            recency = tenure_days
            frequency = 0
            monetary = 0.0
            aov = 0.0
            avg_items_per_order = 0.0
            discount_usage_pct = 0.0
            cat_entropy = 0.0
            cat_unique_count = 0
            preferred_cat = "None"
            avg_time_between_purchases = tenure_days

        # Frequency per month
        monthly_freq = round(frequency / (tenure_days / 30.42), 2)
        repeat_customer = 1 if frequency > 1 else 0
        revenue_contrib = round((monetary / total_system_revenue) * 100, 4) if total_system_revenue > 0 else 0.0
        
        # Refund rate
        total_orders_attempted = len(cust_t_all)
        refund_orders_count = len(cust_t_ref)
        return_refund_rate = round((refund_orders_count / total_orders_attempted) * 100, 2) if total_orders_attempted > 0 else 0.0

        # Interactions Features
        cust_inter = i_by_cust.get(cid, pd.DataFrame())
        if not cust_inter.empty:
            total_interactions = len(cust_inter)
            last_i_date = cust_inter["interaction_date"].max()
            days_since_last_interaction = max(0, (as_of_date - last_i_date).days)
            avg_engagement_val = round(float(cust_inter["engagement_value"].mean()), 2)
            
            email_clicks = len(cust_inter[cust_inter["interaction_type"] == "Email Click"])
            email_opens = len(cust_inter[cust_inter["interaction_type"] == "Email Open"])
            digital_activity = len(cust_inter[cust_inter["interaction_type"].isin(["Website Visit", "App Session", "Product View"])])
            campaign_responses = len(cust_inter[cust_inter["interaction_type"] == "Campaign Response"])
            campaign_response_rate = round((campaign_responses / total_interactions) * 100, 2)
            support_contacts = len(cust_inter[cust_inter["interaction_type"] == "Support Contact"])
        else:
            total_interactions = 0
            days_since_last_interaction = tenure_days
            avg_engagement_val = 0.0
            email_clicks = 0
            email_opens = 0
            digital_activity = 0
            campaign_responses = 0
            campaign_response_rate = 0.0
            support_contacts = 0

        # Customer Lifetime Value heuristic estimation
        # CLV = AOV * (Frequency per year) * Average Margin (30%) * Estimated Lifespan Years
        annual_freq = monthly_freq * 12
        clv_heuristic = round(aov * annual_freq * 0.35 * 2.5, 2) if frequency > 0 else 0.0
        
        feat_dict = {
            "customer_id": cid,
            "age": c_row.get("age", 35),
            "gender": c_row.get("gender", "Unknown"),
            "location": c_row.get("location", "Unknown"),
            "acquisition_channel": c_row.get("acquisition_channel", "Unknown"),
            "customer_type": c_row.get("customer_type", "Retail"),
            "customer_tenure_days": tenure_days,
            
            # RFM Core
            "recency_days": recency,
            "purchase_frequency": frequency,
            "monetary_value": monetary,
            
            # Spending & Order Behavior
            "avg_order_value": aov,
            "avg_items_per_order": avg_items_per_order,
            "monthly_purchase_frequency": monthly_freq,
            "discount_usage_pct": discount_usage_pct,
            "category_diversity_entropy": cat_entropy,
            "unique_categories_count": cat_unique_count,
            "preferred_category": preferred_cat,
            "avg_days_between_purchases": avg_time_between_purchases,
            "repeat_customer_flag": repeat_customer,
            "revenue_contribution_pct": revenue_contrib,
            "return_refund_rate": return_refund_rate,
            "estimated_clv": clv_heuristic,
            
            # Digital Engagement
            "total_interactions": total_interactions,
            "days_since_last_interaction": days_since_last_interaction,
            "avg_engagement_score": avg_engagement_val,
            "email_clicks_count": email_clicks,
            "email_opens_count": email_opens,
            "digital_activity_count": digital_activity,
            "campaign_response_rate": campaign_response_rate,
            "support_contacts_count": support_contacts
        }
        customer_features.append(feat_dict)
        
    df_features = pd.DataFrame(customer_features)
    
    # Generate Feature Dictionary Metadata
    dictionary = get_feature_dictionary()
    
    return df_features, dictionary

def get_feature_dictionary() -> pd.DataFrame:
    """Returns a rich feature dictionary table explaining each variable."""
    items = [
        {
            "Feature Name": "recency_days",
            "Category": "RFM Core",
            "Formula / Logic": "Days between reference date and most recent completed order",
            "Business Meaning": "How recently the customer purchased. Low recency signals active relationship.",
            "Recommended for ML": "Yes (Core)"
        },
        {
            "Feature Name": "purchase_frequency",
            "Category": "RFM Core",
            "Formula / Logic": "Total count of completed orders during observation period",
            "Business Meaning": "Purchase momentum and transaction habits.",
            "Recommended for ML": "Yes (Core)"
        },
        {
            "Feature Name": "monetary_value",
            "Category": "RFM Core",
            "Formula / Logic": "Sum of net completed transaction amounts",
            "Business Meaning": "Total direct financial value generated by customer.",
            "Recommended for ML": "Yes (Core)"
        },
        {
            "Feature Name": "avg_order_value",
            "Category": "Spending Behavior",
            "Formula / Logic": "Monetary Value / Purchase Frequency",
            "Business Meaning": "Basket size per purchase. Distinguishes high-ticket buyers from micro-buyers.",
            "Recommended for ML": "Yes (High Value)"
        },
        {
            "Feature Name": "discount_usage_pct",
            "Category": "Price Sensitivity",
            "Formula / Logic": "Total discount amount / Gross spend * 100",
            "Business Meaning": "Price elasticity and deal sensitivity. Identifies promo-chasers.",
            "Recommended for ML": "Yes (High Value)"
        },
        {
            "Feature Name": "category_diversity_entropy",
            "Category": "Product Affinity",
            "Formula / Logic": "Shannon Entropy of purchased categories: -sum(p * log2(p))",
            "Business Meaning": "Cross-category adoption. High entropy indicates broad brand exploration.",
            "Recommended for ML": "Yes (Exploratory)"
        },
        {
            "Feature Name": "monthly_purchase_frequency",
            "Category": "Velocity",
            "Formula / Logic": "Frequency / (Tenure Days / 30.42)",
            "Business Meaning": "Tenure-normalized purchasing cadence.",
            "Recommended for ML": "Optional"
        },
        {
            "Feature Name": "return_refund_rate",
            "Category": "Risk & Quality",
            "Formula / Logic": "Refunded Orders / Total Attempted Orders * 100",
            "Business Meaning": "Customer satisfaction or product sizing friction indicator.",
            "Recommended for ML": "Optional"
        },
        {
            "Feature Name": "total_interactions",
            "Category": "Omnichannel Engagement",
            "Formula / Logic": "Count of all recorded non-transactional touchpoints",
            "Business Meaning": "Overall brand attention and touchpoint activity.",
            "Recommended for ML": "Yes (Engagement)"
        },
        {
            "Feature Name": "digital_activity_count",
            "Category": "Omnichannel Engagement",
            "Formula / Logic": "Sum of website visits, app sessions, and product views",
            "Business Meaning": "Active digital window shopping and product discovery intent.",
            "Recommended for ML": "Yes (Engagement)"
        },
        {
            "Feature Name": "campaign_response_rate",
            "Category": "Marketing Responsiveness",
            "Formula / Logic": "Campaign Responses / Total Interactions * 100",
            "Business Meaning": "Propensity to act on outbound marketing communications.",
            "Recommended for ML": "Yes (Engagement)"
        },
        {
            "Feature Name": "days_since_last_interaction",
            "Category": "Engagement Recency",
            "Formula / Logic": "Days from reference date to last interaction of any type",
            "Business Meaning": "Early indicator of digital drop-off before transaction churn.",
            "Recommended for ML": "Yes (Retention)"
        },
        {
            "Feature Name": "customer_tenure_days",
            "Category": "Lifecycle",
            "Formula / Logic": "Days from account registration to reference date",
            "Business Meaning": "Relationship longevity with the company.",
            "Recommended for ML": "Optional"
        },
        {
            "Feature Name": "estimated_clv",
            "Category": "Financial Forecast",
            "Formula / Logic": "AOV * Annualized Frequency * 35% Margin * 2.5 Yr Lifespan",
            "Business Meaning": "Heuristic forward-looking customer lifetime value.",
            "Recommended for ML": "Benchmark only"
        }
    ]
    return pd.DataFrame(items)
