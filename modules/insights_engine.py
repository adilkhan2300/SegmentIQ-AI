"""
SegmentIQ AI - Automated AI Insights and Pattern Discovery Engine
Discovers empirical patterns, Pareto distributions, engagement gaps, and segment anomalies
with supporting metrics, actionable interpretations, and limitation guardrails.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

def discover_automated_insights(
    df_clustered: pd.DataFrame,
    df_transactions: pd.DataFrame,
    cluster_summary_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Scans the clustered dataset and summary statistics to surface high-impact business insights.
    """
    insights = []
    
    if df_clustered.empty:
        return insights
        
    total_customers = len(df_clustered)
    total_revenue = df_clustered["monetary_value"].sum() if "monetary_value" in df_clustered.columns else 0
    
    # 1. Pareto Revenue Concentration Analysis
    if total_revenue > 0:
        sorted_rev = df_clustered["monetary_value"].sort_values(ascending=False)
        top_20_pct_count = max(1, int(0.20 * total_customers))
        top_20_rev = sorted_rev.iloc[:top_20_pct_count].sum()
        top_20_share = round((top_20_rev / total_revenue) * 100, 1)
        
        # Identify top segment by revenue
        top_rev_seg = cluster_summary_df.sort_values(by="revenue_share_pct", ascending=False).iloc[0]
        
        insights.append({
            "category": "Revenue Concentration",
            "title": f"Extreme Revenue Skew: Top 20% of Customers Drive {top_20_share}% of Total Revenue",
            "segment": f"{top_rev_seg['segment_code']}",
            "supporting_metrics": (
                f"Top 20% Count: {top_20_pct_count:,} accounts | "
                f"Revenue Generated: ${top_20_rev:,.2f} ({top_20_share}% share) | "
                f"Leading Segment ({top_rev_seg['segment_code']}) alone accounts for {top_rev_seg['revenue_share_pct']}% of total revenue."
            ),
            "interpretation": (
                "A classical Pareto power-law distribution is active. The business relies heavily on an elite core cohort. "
                "Any attrition in this group poses a disproportionate threat to top-line performance."
            ),
            "recommended_action": (
                "Implement a white-glove VIP protection protocol. Offer premium early-access privileges and prioritize customer service "
                "routing over mass generic discounting."
            ),
            "limitations": "Calculated over historical observation window; does not account for forward-looking cohort margin seasonality."
        })

    # Resolve column names dynamically
    freq_col = "purchase_frequency" if "purchase_frequency" in cluster_summary_df.columns else ("avg_frequency" if "avg_frequency" in cluster_summary_df.columns else None)
    aov_col = "avg_order_value" if "avg_order_value" in cluster_summary_df.columns else ("avg_aov" if "avg_aov" in cluster_summary_df.columns else None)
    mon_col = "monetary_value" if "monetary_value" in cluster_summary_df.columns else ("avg_monetary" if "avg_monetary" in cluster_summary_df.columns else None)
    rec_col = "recency_days" if "recency_days" in cluster_summary_df.columns else ("avg_recency" if "avg_recency" in cluster_summary_df.columns else None)
    disc_col = "discount_usage_pct" if "discount_usage_pct" in cluster_summary_df.columns else ("avg_discount_pct" if "avg_discount_pct" in cluster_summary_df.columns else None)

    # 2. High Frequency with Low Basket Size (Everyday Micro-Shoppers)
    if freq_col and aov_col:
        low_aov_high_freq = cluster_summary_df[
            (cluster_summary_df[freq_col] > cluster_summary_df[freq_col].mean()) &
            (cluster_summary_df[aov_col] < cluster_summary_df[aov_col].mean() * 0.75)
        ]
        if not low_aov_high_freq.empty:
            seg = low_aov_high_freq.iloc[0]
            insights.append({
                "category": "Basket Optimization",
                "title": f"Frequent Everyday Buyers Constrained by Low Basket Size ({seg['segment_code']})",
                "segment": f"{seg['segment_code']}",
                "supporting_metrics": (
                    f"Purchase Frequency: {seg[freq_col]:.1f} orders (Above avg) | "
                    f"Average Order Value: ${seg[aov_col]:.2f} (Below avg) | "
                    f"Customer Volume: {int(seg['customer_count'])} customers."
                ),
                "interpretation": (
                    "Customers have established strong purchasing habits but maintain strictly constrained basket sizes, "
                    "likely ordering only immediate essentials."
                ),
                "recommended_action": (
                    "Deploy progressive threshold promotions (e.g. 'Add $12 to earn 150 bonus loyalty points or free gift') "
                    "and smart 'Frequently Bought Together' cart recommendations."
                ),
                "limitations": "Elasticity to order minimums must be tested to ensure orders are not suppressed."
            })

    # 3. High-Value At-Risk Dormancy (VIPs with High Recency)
    if mon_col and rec_col:
        high_val_dormant = cluster_summary_df[
            (cluster_summary_df[mon_col] > cluster_summary_df[mon_col].mean() * 1.2) &
            (cluster_summary_df[rec_col] > 80)
        ]
        if not high_val_dormant.empty:
            seg = high_val_dormant.iloc[0]
            insights.append({
                "category": "Churn Prevention",
                "title": f"High-Value Capital at Risk: Premium Cohort Showing Extended Inactivity ({seg['segment_code']})",
                "segment": f"{seg['segment_code']}",
                "supporting_metrics": (
                    f"Historical Avg Spend: ${seg[mon_col]:,.2f} | "
                    f"Lapsed Recency: {seg[rec_col]:.0f} days since last purchase | "
                    f"Impacted Customers: {int(seg['customer_count'])} accounts."
                ),
                "interpretation": (
                    "Accounts with high demonstrated willingness-to-pay are experiencing substantial silence. "
                    "This divergence often signals competitor switching, friction in their last fulfillment experience, or lifecycle disengagement."
                ),
                "recommended_action": (
                    "Trigger an executive win-back campaign featuring a direct personal outreach or high-value account credit "
                    "contingent on taking a short feedback questionnaire."
                ),
                "limitations": "Lapse may partially reflect natural multi-year product lifecycles if high-ticket durable items were purchased."
            })

    # 4. Extreme Discount Dependency (Promo Chasers)
    if disc_col and freq_col:
        promo_seg = cluster_summary_df[cluster_summary_df[disc_col] > 20.0]
        if not promo_seg.empty:
            seg = promo_seg.iloc[0]
            insights.append({
                "category": "Margin Preservation",
                "title": f"Margin Cannibalization Risk: Deep Discount Dependency in {seg['segment_code']}",
                "segment": f"{seg['segment_code']}",
                "supporting_metrics": (
                    f"Average Discount Rate: {seg[disc_col]:.1f}% | "
                    f"Orders Placed: {seg[freq_col]:.1f} | "
                    f"Segment Count: {int(seg['customer_count'])} customers."
                ),
                "interpretation": (
                    "This segment exhibits near-complete price elasticity and is trained to wait for promotional markdowns, "
                    "eroding enterprise operating margins on standard inventory."
                ),
                "recommended_action": (
                    "Restrict promotional communications for this segment strictly to clearance inventory and end-of-season closeouts. "
                    "Do not offer sitewide coupon codes during regular retail periods."
                ),
                "limitations": "Reducing discounts may lower immediate transaction frequency for this specific cohort."
            })

    # 5. High Engagement with Zero/Single Purchases (Window Shoppers)
    window_shoppers = df_clustered[
        (df_clustered["total_interactions"] > df_clustered["total_interactions"].median()) &
        (df_clustered["purchase_frequency"] <= 1)
    ]
    if len(window_shoppers) > 20:
        insights.append({
            "category": "Conversion Opportunity",
            "title": f"Unmonetized Traffic: {len(window_shoppers)} Customers High in Engagement but Low in Purchases",
            "segment": "Cross-Segment Digital Browsers",
            "supporting_metrics": (
                f"Identified Pool: {len(window_shoppers)} customers | "
                f"Average Interactions: {window_shoppers['total_interactions'].mean():.1f} touchpoints | "
                f"Digital Activity Count: {window_shoppers['digital_activity_count'].mean():.1f} visits/sessions."
            ),
            "interpretation": (
                "Prospective buyers are deeply inspecting product pages and reading communications, but hesitation at checkout "
                "prevents repeat transaction completion. This reflects interest paired with perceived barrier to entry."
            ),
            "recommended_action": (
                "Deploy reassurance triggers: Money-back guarantees, verified customer social proof badges, and limited-time "
                "free shipping thresholds on their first or second order."
            ),
            "limitations": "Interaction telemetry does not capture specific on-site technical errors or checkout friction details."
        })

    # 6. Single-Purchase Drop-Off Rate
    single_buyers = (df_clustered["purchase_frequency"] == 1).sum()
    single_pct = round((single_buyers / total_customers) * 100, 1)
    if single_pct > 25.0:
        insights.append({
            "category": "Onboarding & Retention",
            "title": f"First-to-Second Purchase Chasm: {single_pct}% of Customers Never Placed a Second Order",
            "segment": "Single-Purchase Cohort",
            "supporting_metrics": (
                f"Single-Order Customers: {single_buyers:,} accounts ({single_pct}% of total customer base) | "
                f"Repeat Purchase Rate: {100 - single_pct:.1f}%."
            ),
            "interpretation": (
                "The steepest customer drop-off occurs between order 1 and order 2. "
                "Acquisition costs (CAC) are squandered when customers churn after a single trial purchase."
            ),
            "recommended_action": (
                "Implement an automated 30-day Post-Purchase Onboarding Journey providing usage guides, customer support check-ins, "
                "and a time-bound cross-category recommendation for order #2."
            ),
            "limitations": "Customers registered within the past 30 days may still be in their natural initial purchase cycle."
        })

    return insights
