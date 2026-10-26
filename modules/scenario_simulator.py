"""
SegmentIQ AI - Interactive Marketing Scenario Simulator
Models what-if financial projections for retention boosts, budget reallocations,
discount adjustments, and browsing conversion campaigns.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

def run_scenario_simulation(
    df_clustered: pd.DataFrame,
    retention_boost_pct: float = 15.0,
    repeat_rate_boost_pct: float = 10.0,
    discount_reduction_pct: float = 20.0,
    window_shopper_conversion_pct: float = 5.0,
    marketing_budget: float = 15000.0,
    gross_margin_pct: float = 38.0
) -> Dict[str, Any]:
    """
    Computes baseline actuals vs scenario projected financial performance.
    """
    if df_clustered.empty:
        return {}
        
    total_customers = len(df_clustered)
    base_orders = int(df_clustered["purchase_frequency"].sum()) if "purchase_frequency" in df_clustered.columns else 0
    base_revenue = float(df_clustered["monetary_value"].sum()) if "monetary_value" in df_clustered.columns else 0.0
    base_aov = base_revenue / base_orders if base_orders > 0 else 85.0
    
    # 1. Retention Boost on At-Risk Customers
    # Identify at-risk: recency > 90 days or segment with high recency
    at_risk_mask = df_clustered["recency_days"] > 80
    at_risk_count = at_risk_mask.sum()
    at_risk_saved = int(round(at_risk_count * (retention_boost_pct / 100.0)))
    sim_retention_orders = int(round(at_risk_saved * 1.5))
    sim_retention_rev = sim_retention_orders * base_aov
    
    # 2. Repeat Rate Boost on Single-Order Customers
    single_order_count = (df_clustered["purchase_frequency"] == 1).sum()
    converted_to_repeat = int(round(single_order_count * (repeat_rate_boost_pct / 100.0)))
    sim_repeat_orders = converted_to_repeat * 1  # 1 additional order each
    sim_repeat_rev = sim_repeat_orders * (base_aov * 0.9)
    
    # 3. Discount Optimization (Margin Recovery)
    # Recovering gross margin from promo-heavy customers by reducing discount leakage
    total_discounts_given = 0.0
    if "discount_usage_pct" in df_clustered.columns:
        # Approximate discount dollars = revenue * (discount_usage_pct / (100 - discount_usage_pct))
        approx_discount_dollars = (base_revenue * (df_clustered["discount_usage_pct"].mean() / 100.0))
        margin_recovered = approx_discount_dollars * (discount_reduction_pct / 100.0)
    else:
        margin_recovered = 0.0
        
    # 4. Converting Engaged Window Shoppers
    window_count = ((df_clustered["total_interactions"] > 15) & (df_clustered["purchase_frequency"] <= 1)).sum()
    converted_browsers = int(round(window_count * (window_shopper_conversion_pct / 100.0)))
    sim_browser_orders = converted_browsers * 1
    sim_browser_rev = sim_browser_orders * (base_aov * 0.8)
    
    # Aggregate Scenario Totals
    incremental_orders = sim_retention_orders + sim_repeat_orders + sim_browser_orders
    incremental_gross_revenue = sim_retention_rev + sim_repeat_rev + sim_browser_rev
    
    scenario_total_orders = base_orders + incremental_orders
    scenario_total_revenue = base_revenue + incremental_gross_revenue
    
    # Financial Margins & ROI
    baseline_gross_profit = base_revenue * (gross_margin_pct / 100.0)
    incremental_gross_profit = (incremental_gross_revenue * (gross_margin_pct / 100.0)) + margin_recovered
    scenario_gross_profit = baseline_gross_profit + incremental_gross_profit
    
    net_incremental_profit = incremental_gross_profit - marketing_budget
    campaign_roi_pct = round((net_incremental_profit / marketing_budget) * 100, 1) if marketing_budget > 0 else 0.0
    
    return {
        "baseline": {
            "total_customers": total_customers,
            "orders": base_orders,
            "revenue": round(base_revenue, 2),
            "aov": round(base_aov, 2),
            "gross_profit": round(baseline_gross_profit, 2)
        },
        "scenario": {
            "marketing_budget": marketing_budget,
            "total_orders": scenario_total_orders,
            "total_revenue": round(scenario_total_revenue, 2),
            "gross_profit": round(scenario_gross_profit, 2),
            "incremental_orders": incremental_orders,
            "incremental_revenue": round(incremental_gross_revenue, 2),
            "margin_recovered_from_discounts": round(margin_recovered, 2),
            "incremental_gross_profit": round(incremental_gross_profit, 2),
            "net_incremental_profit": round(net_incremental_profit, 2),
            "projected_roi_pct": campaign_roi_pct,
            "retention_reactivations": at_risk_saved,
            "repeat_conversions": converted_to_repeat,
            "browser_conversions": converted_browsers
        }
    }
