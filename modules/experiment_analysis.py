"""
SegmentIQ AI - Campaign Experimentation and A/B Testing Analysis Module
Calculates sample sizes, conversion rates, statistical hypothesis testing (Z-test, p-values),
confidence intervals, revenue lift, and segment-level treatment effects.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple

def analyze_ab_experiment(
    df_experiments: pd.DataFrame,
    control_variant: str = "Variant_A_Control",
    treatment_variant: str = "Variant_B_Personalized",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Evaluates conversion lift and revenue difference between A/B variants.
    Computes two-proportion z-test, p-value, 95% CI, and Welch's t-test for revenue.
    """
    if df_experiments is None or df_experiments.empty:
        return {"error": "Experiment dataset is empty."}
        
    ctrl = df_experiments[df_experiments["variant"] == control_variant]
    treat = df_experiments[df_experiments["variant"] == treatment_variant]
    
    n_a = len(ctrl)
    n_b = len(treat)
    
    if n_a < 10 or n_b < 10:
        return {"error": f"Insufficient sample size: Control={n_a}, Treatment={n_b}. Require at least 10 each."}
        
    # Conversions
    conv_a = int(ctrl["converted"].sum())
    conv_b = int(treat["converted"].sum())
    
    cr_a = conv_a / n_a
    cr_b = conv_b / n_b
    
    abs_lift = cr_b - cr_a
    rel_lift = (abs_lift / cr_a * 100) if cr_a > 0 else 0.0
    
    # Two-proportion z-test
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    
    z_score = (cr_b - cr_a) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Confidence Interval for proportion difference
    z_crit = stats.norm.ppf((1 + confidence_level) / 2)
    se_diff = np.sqrt((cr_a * (1 - cr_a) / n_a) + (cr_b * (1 - cr_b) / n_b))
    ci_lower = abs_lift - z_crit * se_diff
    ci_upper = abs_lift + z_crit * se_diff
    
    # Revenue comparison
    rev_a = float(ctrl["order_value"].sum()) if "order_value" in ctrl.columns else 0.0
    rev_b = float(treat["order_value"].sum()) if "order_value" in treat.columns else 0.0
    rpu_a = rev_a / n_a
    rpu_b = rev_b / n_b
    
    # T-test for order value
    t_stat, t_pval = stats.ttest_ind(
        treat["order_value"], ctrl["order_value"], equal_var=False
    )
    
    is_significant = bool(p_value < (1.0 - confidence_level))
    
    return {
        "control_name": control_variant,
        "treatment_name": treatment_variant,
        "n_control": n_a,
        "n_treatment": n_b,
        "conversions_control": conv_a,
        "conversions_treatment": conv_b,
        "conversion_rate_control": round(cr_a * 100, 2),
        "conversion_rate_treatment": round(cr_b * 100, 2),
        "absolute_lift_pct_points": round(abs_lift * 100, 2),
        "relative_lift_pct": round(rel_lift, 2),
        "z_score": round(float(z_score), 3),
        "p_value": round(float(p_value), 5),
        "ci_lower_pct": round(ci_lower * 100, 2),
        "ci_upper_pct": round(ci_upper * 100, 2),
        "is_statistically_significant": is_significant,
        "revenue_control": round(rev_a, 2),
        "revenue_treatment": round(rev_b, 2),
        "rev_per_user_control": round(rpu_a, 2),
        "rev_per_user_treatment": round(rpu_b, 2),
        "revenue_t_pvalue": round(float(t_pval), 5)
    }

def simulate_hypothetical_campaign(
    target_segment_size: int,
    channel: str,
    budget: float,
    baseline_conversion_rate: float,
    projected_lift_pct: float,
    average_order_value: float,
    gross_margin_pct: float = 35.0
) -> Dict[str, Any]:
    """
    Calculates reach, projected conversions, gross revenue, contribution margin,
    and estimated Campaign ROI for hypothetical scenario planning.
    """
    # Estimated channel cost per contact
    cost_per_contact_map = {
        "Email": 0.05,
        "SMS": 0.12,
        "Push Notification": 0.02,
        "Paid Retargeting Ad": 0.85,
        "Direct Mail / Catalog": 2.50
    }
    unit_cost = cost_per_contact_map.get(channel, 0.10)
    
    max_reach_by_budget = int(budget / unit_cost) if unit_cost > 0 else target_segment_size
    customers_reached = min(target_segment_size, max_reach_by_budget)
    campaign_spend = min(budget, customers_reached * unit_cost)
    
    # Conversions
    baseline_orders = customers_reached * (baseline_conversion_rate / 100.0)
    projected_cr = (baseline_conversion_rate / 100.0) * (1 + projected_lift_pct / 100.0)
    projected_orders = customers_reached * projected_cr
    incremental_orders = projected_orders - baseline_orders
    
    # Financials
    projected_revenue = projected_orders * average_order_value
    incremental_revenue = incremental_orders * average_order_value
    gross_profit = incremental_revenue * (gross_margin_pct / 100.0)
    net_profit = gross_profit - campaign_spend
    
    roi_pct = round((net_profit / campaign_spend * 100), 1) if campaign_spend > 0 else 0.0
    
    return {
        "target_audience_size": target_segment_size,
        "customers_reached": customers_reached,
        "reach_percentage": round((customers_reached / target_segment_size * 100), 1) if target_segment_size > 0 else 0,
        "campaign_spend": round(campaign_spend, 2),
        "projected_orders": int(round(projected_orders)),
        "incremental_orders": int(round(incremental_orders)),
        "projected_revenue": round(projected_revenue, 2),
        "incremental_revenue": round(incremental_revenue, 2),
        "gross_contribution_profit": round(gross_profit, 2),
        "net_profit": round(net_profit, 2),
        "estimated_roi_pct": roi_pct
    }

def calculate_sample_size_requirement(
    baseline_cr: float = 0.08,
    mde_relative: float = 0.20,
    power: float = 0.80,
    significance: float = 0.05
) -> Dict[str, Any]:
    """
    Calculates sample size requirement per variant for two-proportion A/B test.
    """
    p1 = baseline_cr
    p2 = baseline_cr * (1 + mde_relative)
    p_bar = (p1 + p2) / 2.0
    
    z_alpha = stats.norm.ppf(1 - significance / 2.0)
    z_beta = stats.norm.ppf(power)
    
    numerator = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    
    n_per_variant = int(np.ceil(numerator / denominator)) if denominator > 0 else 1000
    
    return {
        "baseline_conversion_rate": p1,
        "minimum_detectable_effect_relative": mde_relative,
        "statistical_power": power,
        "significance_alpha": significance,
        "required_sample_per_variant": n_per_variant,
        "total_sample_required": n_per_variant * 2
    }

