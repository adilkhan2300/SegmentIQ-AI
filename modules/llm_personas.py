"""
SegmentIQ AI - LLM & Rule-Based Customer Persona Generator
Transforms numerical cluster metrics into rich, behavioral customer personas.
Supports OpenAI API (with structured JSON validation) and advanced Rule-Based fallback.
"""

import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd

def generate_personas(
    cluster_summary_df: pd.DataFrame,
    overall_stats: Dict[str, Any],
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Generates customer personas for all segments in cluster_summary_df.
    Attempts OpenAI API if api_key provided; otherwise runs rule-based generator.
    """
    # Check if OpenAI is available and key is provided
    use_llm = bool(api_key and api_key.strip() and not api_key.startswith("sk-placeholder"))
    
    if use_llm:
        try:
            return _generate_personas_openai(cluster_summary_df, overall_stats, api_key.strip(), model_name)
        except Exception as e:
            print(f"OpenAI persona generation error: {e}. Falling back to Rule-Based Engine.")
            result = _generate_personas_rule_based(cluster_summary_df, overall_stats)
            result["llm_error"] = str(e)
            result["generator_type"] = "Rule-Based Engine (API Fallback)"
            return result
    else:
        return _generate_personas_rule_based(cluster_summary_df, overall_stats)

def _generate_personas_openai(
    cluster_summary_df: pd.DataFrame,
    overall_stats: Dict[str, Any],
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """Calls OpenAI chat completion with strict schema prompting."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    # Prepare sanitized summary payload
    summary_records = cluster_summary_df.to_dict(orient="records")
    
    system_prompt = (
        "You are an elite Chief Customer Intelligence Officer and Behavioral Marketing Scientist. "
        "Your task is to interpret unsupervised customer clustering results and generate executive-grade "
        "customer personas. "
        "Strict rules:\n"
        "1. Never invent statistics or quote numbers not in the input data.\n"
        "2. Clearly label all needs, desires, and psychological drivers as 'Hypotheses'.\n"
        "3. Present correlations as observations, not proven causal facts.\n"
        "4. Return strictly valid JSON matching the requested structure."
    )
    
    user_prompt = f"""
Here is the baseline overall business benchmark:
- Population Total Customers: {overall_stats.get('total_customers', 'N/A')}
- Baseline AOV: ${overall_stats.get('avg_aov', 'N/A')}
- Baseline Frequency: {overall_stats.get('avg_freq', 'N/A')} orders
- Baseline Recency: {overall_stats.get('avg_recency', 'N/A')} days

Here are the cluster summary records for each discovered segment:
{json.dumps(summary_records, indent=2)}

Generate a structured JSON object with key "personas" mapping each segment_code to an object with:
- "persona_title": Evocative, professional title (e.g., 'High-Value VIP Loyalists', 'High-Engagement Window Shoppers')
- "short_description": 2-3 sentence executive synopsis
- "dominant_behaviors": List of 3 key observed purchase & activity traits
- "spending_profile": Description of monetary value and AOV relative to baseline
- "engagement_pattern": Description of interaction velocity and touchpoints
- "hypothesized_needs": List of 2-3 behavioral motivations (explicitly stated as hypotheses)
- "retention_barriers": List of 2 potential friction points or churn risks
- "empirical_evidence": Concrete metrics from the data that justify this persona
- "confidence_assessment": Data sufficiency evaluation (High / Moderate / Low) with rationale
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3
    )
    
    raw_content = response.choices[0].message.content
    parsed = json.loads(raw_content)
    
    return {
        "generator_type": f"OpenAI LLM ({model_name})",
        "personas": parsed.get("personas", {}),
        "raw_response": raw_content
    }

def _generate_personas_rule_based(
    cluster_summary_df: pd.DataFrame,
    overall_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Statistically grounded rule-based persona engine.
    Calculates z-scores and relative deltas to baseline for automated persona synthesis.
    """
    personas = {}
    
    # Calculate population baselines
    base_aov = overall_stats.get("avg_aov", cluster_summary_df["avg_order_value"].mean() if "avg_order_value" in cluster_summary_df.columns else 100)
    base_freq = overall_stats.get("avg_freq", cluster_summary_df["purchase_frequency"].mean() if "purchase_frequency" in cluster_summary_df.columns else 5)
    base_rec = overall_stats.get("avg_recency", cluster_summary_df["recency_days"].mean() if "recency_days" in cluster_summary_df.columns else 45)
    base_disc = cluster_summary_df["discount_usage_pct"].mean() if "discount_usage_pct" in cluster_summary_df.columns else 10
    base_inter = cluster_summary_df["total_interactions"].mean() if "total_interactions" in cluster_summary_df.columns else 15
    
    for _, row in cluster_summary_df.iterrows():
        code = str(row["segment_code"])
        size = int(row.get("customer_count", 0))
        pct_cust = row.get("pct_of_customers", 0.0)
        rev_share = row.get("revenue_share_pct", 0.0)
        
        rec = row.get("recency_days", base_rec)
        freq = row.get("purchase_frequency", base_freq)
        mon = row.get("monetary_value", 0.0)
        aov = row.get("avg_order_value", base_aov)
        disc = row.get("discount_usage_pct", base_disc)
        inter = row.get("total_interactions", base_inter)
        
        # Relative ratios
        aov_ratio = aov / (base_aov + 1e-5)
        freq_ratio = freq / (base_freq + 1e-5)
        rec_ratio = rec / (base_rec + 1e-5)
        disc_ratio = disc / (base_disc + 1e-5)
        inter_ratio = inter / (base_inter + 1e-5)
        
        # Rule classification logic
        if code == "Noise / Outliers":
            title = "Algorithmic Outliers & Fringe Behaviors"
            desc = "Customers with anomalous or atypical multi-dimensional vectors who do not conform cleanly to dense core behavioral clusters."
            hypo_needs = [
                "Hypothesis: May include extreme edge-case wholesale buyers or anomalous test accounts.",
                "Hypothesis: Requires individual manual transaction review before automated marketing enrollment."
            ]
            barriers = ["High behavioral variance makes standard messaging ineffective."]
            conf = "Low (High variance / small sample)"
        elif aov_ratio > 1.35 and freq_ratio >= 1.0:
            title = "High-Value VIP Champions"
            desc = "High-spending, loyal brand advocates generating an outsized share of total enterprise revenue with minimal price sensitivity."
            hypo_needs = [
                "Hypothesis: Values premium white-glove treatment, early product access, and dedicated support over price promotions.",
                "Hypothesis: Retains strong brand affinity when recognized with tier-based VIP status."
            ]
            barriers = ["Dissatisfaction with service delays or feeling commoditized by generic mass discounts."]
            conf = "High (Robust frequency & monetary density)"
        elif aov_ratio > 1.4 and freq_ratio < 0.7:
            title = "Occasion-Driven Premium Buyers"
            desc = "Discerning shoppers who make infrequent, high-ticket purchases, often clustered around major holidays, milestones, or gifting events."
            hypo_needs = [
                "Hypothesis: Highly research-oriented during specific buying windows; requires confidence in quality and warranty.",
                "Hypothesis: Responsive to seasonal catalogs and curated luxury bundles."
            ]
            barriers = ["Prolonged inactivity between seasonal cycles risks complete brand defection."]
            conf = "Moderate (High monetary weight, lower purchase count)"
        elif freq_ratio >= 1.25 and aov_ratio <= 1.0 and disc_ratio < 1.3:
            title = "Frequent Everyday Essentials Buyers"
            desc = "Reliable, regular repeat purchasers with moderate basket sizes who rely on the brand for routine replenishment."
            hypo_needs = [
                "Hypothesis: Prioritizes frictionless checkout, subscription replenishment, and reliable delivery times.",
                "Hypothesis: High receptivity to automated reorder reminders and loyalty point accrual."
            ]
            barriers = ["Checkout friction or inventory stockouts will quickly redirect them to competitors."]
            conf = "High (High transaction frequency across extended tenure)"
        elif disc_ratio >= 1.3 or disc > 20.0:
            title = "Price-Sensitive Discount Seekers"
            desc = "Deal-oriented consumers whose purchasing cadence is strictly triggered by promotional sales, flash deals, and clearance events."
            hypo_needs = [
                "Hypothesis: Motivated by the psychological reward of perceived savings and promo codes.",
                "Hypothesis: High cross-shopping across rival discount platforms."
            ]
            barriers = ["Extremely low willingness to purchase at full MSRP; high margin dilution risk."]
            conf = "High (Clear correlation between discounts and order timing)"
        elif rec_ratio > 1.5 or rec > 110:
            title = "At-Risk Inactive / Dormant Customers"
            desc = "Previously active purchasers whose recency has lapsed significantly, indicating declining relationship momentum and high churn risk."
            hypo_needs = [
                "Hypothesis: May have experienced product dissatisfaction, life-stage transition, or switched to a rival vendor.",
                "Hypothesis: Requires a compelling reactivation hook to re-establish purchase habits."
            ]
            barriers = ["Email unsubscribe risk and communication fatigue if bombarded with generic messages."]
            conf = "High (Strong recency lapse observed across the cohort)"
        elif inter_ratio > 1.2 and freq < 2:
            title = "Engaged Digital Window Shoppers"
            desc = "Digitally active visitors with high browsing and interaction rates who exhibit significant interest but hesitate at final checkout."
            hypo_needs = [
                "Hypothesis: Requires trust signals, social proof, customer reviews, or a first-order incentive to convert.",
                "Hypothesis: May be price checking or waiting for an optimal buying trigger."
            ]
            barriers = ["Cart abandonment friction, unexpected shipping fees, or indecision at point of purchase."]
            conf = "High (Abundant digital interaction logs with low transaction conversion)"
        else:
            title = f"Emerging Core Segment ({code})"
            desc = f"Mainstream customer base with balanced purchasing frequency and moderate engagement levels across the product catalog."
            hypo_needs = [
                "Hypothesis: Seeking balanced product discovery and consistent product reliability.",
                "Hypothesis: Responsive to category cross-selling based on recent browsing."
            ]
            barriers = ["Lack of differentiation could cause slow migration to alternative brands."]
            conf = "Moderate (Balanced behavioral metrics)"
            
        personas[code] = {
            "persona_title": title,
            "short_description": desc,
            "dominant_behaviors": [
                f"Purchase frequency of {freq:.1f} orders vs benchmark {base_freq:.1f}",
                f"Average Order Value of ${aov:.2f} vs benchmark ${base_aov:.2f}",
                f"Average Recency of {rec:.0f} days since last purchase"
            ],
            "spending_profile": f"Generates {rev_share:.1f}% of total enterprise revenue across {pct_cust:.1f}% of customer base (Avg spend: ${mon:,.2f}).",
            "engagement_pattern": f"{inter:.1f} average interactions touchpoints; discount rate averaging {disc:.1f}%.",
            "hypothesized_needs": hypo_needs,
            "retention_barriers": barriers,
            "empirical_evidence": f"Recency: {rec:.0f}d | Freq: {freq:.1f} | AOV: ${aov:.2f} | Discount: {disc:.1f}% | Size: {size} customers",
            "confidence_assessment": conf
        }
        
    return {
        "generator_type": "Rule-Based Behavioral Persona Engine",
        "personas": personas
    }
