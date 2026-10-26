"""
SegmentIQ AI - Differentiated Marketing Strategy Generator
Generates segment-specific marketing campaigns, lifecycle strategies, and rigorous 8-point business justifications.
Avoids blanket discounts and ensures empirical alignment with behavioral data.
"""

from typing import Dict, Any, List
import pandas as pd

def generate_marketing_strategies(
    cluster_summary_df: pd.DataFrame,
    personas_dict: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Synthesizes tailored marketing strategies and lifecycle roadmaps for each segment.
    """
    strategies = {}
    personas = personas_dict.get("personas", {})
    
    for _, row in cluster_summary_df.iterrows():
        code = str(row["segment_code"])
        persona = personas.get(code, {})
        title = persona.get("persona_title", f"Segment {code}")
        
        rec = row.get("recency_days", 45)
        freq = row.get("purchase_frequency", 4)
        aov = row.get("avg_order_value", 80)
        disc = row.get("discount_usage_pct", 10)
        inter = row.get("total_interactions", 15)
        size = int(row.get("customer_count", 0))
        
        # Determine archetype & tailored strategy
        if "VIP" in title or (aov > 140 and freq >= 5):
            lifecycle_stage = "VIP Loyalty Retention & Advocacy"
            objective = "Maximize Net Lifetime Value & Cultivate Brand Ambassadors"
            channel = "Direct VIP Concierge Email & Personalized In-App Experience"
            core_msg = "Exclusive First Access: Preview our newest luxury arrivals before public launch."
            product_cat = "Premium Electronics & Luxury Apparel"
            content_fmt = "Editorial Lookbook & Personal Invitation Letter"
            cta = "Unlock Your VIP Preview"
            incentive = "No price discount. Complimentary express shipping & dedicated concierge support."
            timing = "Mid-week evening (Tuesday/Thursday, 7 PM local)"
            follow_up = "Post-delivery satisfaction outreach from client success executive."
            
            # 8-point justification
            justification = {
                "segment_name": title,
                "behavioral_evidence": f"High AOV (${aov:.2f}) and sustained frequency ({freq:.1f} orders) with low discount reliance ({disc:.1f}%).",
                "marketing_objective": "Retain top-tier accounts without diluting gross margin with unnecessary coupons.",
                "recommended_action": "Enroll in tiered concierge loyalty circle with non-monetary experiential perks.",
                "expected_direction_of_impact": "Positive increase in retention rate (+12%) and annualized spend (+15%).",
                "kpi_to_monitor": "90-day repurchase rate, Customer Net Promoter Score (NPS), Average Gross Margin.",
                "validation_method": "A/B test: VIP concierge perks vs standard communications on 20% holdout.",
                "confidence_level": "High (Strong purchasing volume and sustained transaction history)."
            }

        elif "Occasion" in title or (aov > 160 and freq < 3):
            lifecycle_stage = "Cross-Sell Acceleration & Lifecycle Nurturing"
            objective = "Shorten Repurchase Cycle & Increase Category Exploration"
            channel = "Triggered Email & Retargeting Paid Social"
            core_msg = "Curated collections crafted for your next milestone."
            product_cat = "Home & Living, Fine Jewelry & Electronics"
            content_fmt = "Gift Guides & Interactive Product Matcher"
            cta = "Explore Milestone Curations"
            incentive = "Free premium gift-wrapping or bundled product add-on with orders over $150."
            timing = "3 weeks prior to major seasonal holidays and customer anniversary month"
            follow_up = "Targeted accessory recommendation 14 days post-purchase."
            
            justification = {
                "segment_name": title,
                "behavioral_evidence": f"High basket size (${aov:.2f}) but low order frequency ({freq:.1f} orders).",
                "marketing_objective": "Bridge the gap between occasional purchases without degrading premium perception.",
                "recommended_action": "Implement seasonal milestone triggers and gift-oriented storytelling.",
                "expected_direction_of_impact": "Increase annual order frequency from 1.5 to 2.2 orders per customer.",
                "kpi_to_monitor": "Annual order frequency, repeat purchase conversion rate within 180 days.",
                "validation_method": "Sequential trigger email vs standard newsletter benchmark.",
                "confidence_level": "Moderate (High monetary stability, lower frequency observation)."
            }

        elif "Everyday" in title or (freq >= 8 and aov <= 75):
            lifecycle_stage = "Automated Replenishment & Habit Formation"
            objective = "Lock in Habitual Repeat Purchases & Increase Basket Size"
            channel = "Mobile App Push & SMS Replenishment Alerts"
            core_msg = "Running low? One-tap reorder your everyday favorites today."
            product_cat = "Health & Beauty, Everyday Essentials, Groceries"
            content_fmt = "Quick-reorder carousel & smart replenishment widget"
            cta = "1-Click Reorder"
            incentive = "Threshold incentive: 'Add $15 to unlock a full-sized complimentary sample.'"
            timing = "Predictive timing based on median inter-purchase interval (~18-24 days)"
            follow_up = "SMS dispatch notification with order tracking link."
            
            justification = {
                "segment_name": title,
                "behavioral_evidence": f"High transaction cadence ({freq:.1f} orders) with modest AOV (${aov:.2f}) and steady recency ({rec:.0f} days).",
                "marketing_objective": "Streamline checkout friction and incentivize marginal basket additions.",
                "recommended_action": "Deploy automated replenish triggers and basket-builder product suggestions.",
                "expected_direction_of_impact": "Increase AOV by 10-18% while sustaining weekly purchasing rhythm.",
                "kpi_to_monitor": "AOV per completed order, 30-day order velocity, mobile app engagement rate.",
                "validation_method": "Multi-arm bandit testing of replenishment notification copy and timing.",
                "confidence_level": "High (Statistically dense purchase frequency data)."
            }

        elif "Discount" in title or (disc >= 18.0):
            lifecycle_stage = "Margin-Protected Promotional Activation"
            objective = "Channel Price Elasticity Toward Clearance & Excess Inventory"
            channel = "Promotional Email Blast & Flash Deal Push Notifications"
            core_msg = "Flash Clearance: 48-Hour Outlet Event with exclusive tiered savings."
            product_cat = "Apparel Clearance & Seasonal Overstock"
            content_fmt = "Countdown timer banner with bold savings badges"
            cta = "Shop Limited Clearance"
            incentive = "Volume-tiered threshold discount: 'Save 15% on $60, or 25% on $100+' to protect margin."
            timing = "Friday lunchtime (12:00 PM) or month-end inventory liquidation"
            follow_up = "Last-chance cart reminder 4 hours prior to promo expiry."
            
            justification = {
                "segment_name": title,
                "behavioral_evidence": f"Elevated discount dependency ({disc:.1f}% discount rate) and high promo sensitivity.",
                "marketing_objective": "Direct deal-seeking behavior to liquidate overstock without cannibalizing full-price margins.",
                "recommended_action": "Use strict volume thresholds ($60/$100) instead of flat site-wide discount codes.",
                "expected_direction_of_impact": "Maintain positive gross contribution margin while satisfying promo demand.",
                "kpi_to_monitor": "Net gross margin per order, promotion redemption rate, clearance inventory turn.",
                "validation_method": "Compare flat 20% discount code vs tiered threshold ($15 off $75) on promo cohorts.",
                "confidence_level": "High (Clear empirical proof of price elasticity)."
            }

        elif "At-Risk" in title or (rec >= 100):
            lifecycle_stage = "Win-Back & Reactivation"
            objective = "Reignite Relationship Momentum & Diagnose Inactivity Friction"
            channel = "Win-Back Email Sequence & Targeted Feedback Survey"
            core_msg = "We've missed you! Tell us how we can serve you better + a welcome-back gesture."
            product_cat = "Customer's past favorite category based on purchase history"
            content_fmt = "Conversational founder note + 1-question feedback poll"
            cta = "Claim Your Welcome Back Credit"
            incentive = "$20 account credit on a purchase of $75 or more (valid for 14 days)."
            timing = "Triggered at 90 days of inactivity, sent Tuesday 10 AM"
            follow_up = "Reminder email 3 days before credit expiration."
            
            justification = {
                "segment_name": title,
                "behavioral_evidence": f"Severe recency lapse ({rec:.0f} days without purchase) following past active transaction history.",
                "marketing_objective": "Re-establish active purchase habit before customer crosses into irreversible churn.",
                "recommended_action": "Deploy timed 2-step win-back sequence with bounded account credit and feedback survey.",
                "expected_direction_of_impact": "Reactivate 8-14% of dormant customer base into active buyers within 30 days.",
                "kpi_to_monitor": "Win-back conversion rate, reactivation CAC, survey completion rate.",
                "validation_method": "Win-back credit email vs simple 'We Miss You' non-incentivized email control.",
                "confidence_level": "High (Pronounced recency divergence from active population)."
            }

        elif "Window" in title or (inter > 20 and freq <= 1):
            lifecycle_stage = "First-Order Conversion & Trust Building"
            objective = "Convert High Browsing Intent into First Completed Purchase"
            channel = "Browse Abandonment Email & Welcome Retargeting Ads"
            core_msg = "Complete your setup: Enjoy free returns and 10% off your first qualifying order."
            product_cat = "Best-Selling Starter Kits & Verified Top-Rated Items"
            content_fmt = "Customer testimonials, UGC videos, and risk-free trial guarantee"
            cta = "Start With Confidence"
            incentive = "10% first-order discount code + 60-day hassle-free money-back guarantee."
            timing = "Triggered 3 hours post-session browsing event"
            follow_up = "Social proof review compilation 48 hours later."
            
            justification = {
                "segment_name": title,
                "behavioral_evidence": f"High digital touchpoints ({inter:.1f} interactions) but near-zero transaction count ({freq:.1f} orders).",
                "marketing_objective": "Lower perceived purchase risk and incentivize first transaction conversion.",
                "recommended_action": "Emphasize trust badges, verified customer reviews, and a bounded first-order incentive.",
                "expected_direction_of_impact": "Improve browse-to-purchase conversion from ~3% to ~7.5%.",
                "kpi_to_monitor": "First-purchase conversion rate, add-to-cart rate, bounce rate.",
                "validation_method": "A/B test: Social proof guarantee messaging vs immediate 10% coupon.",
                "confidence_level": "High (Substantial interaction telemetry confirms active interest)."
            }

        else:
            lifecycle_stage = "Core Customer Development"
            objective = "Expand Category Breadth & Increase Engagement Frequency"
            channel = "Personalized Newsletter & Targeted Product Recommendations"
            core_msg = "Discover handpicked picks tailored to your recent preferences."
            product_cat = "Complementary Accessories & Trending Categories"
            content_fmt = "Personalized dynamic product grid"
            cta = "Discover Recommendations"
            incentive = "Double loyalty reward points on any new product category purchase this month."
            timing = "Saturday morning (9:30 AM local)"
            follow_up = "Follow up with related buying guide."
            
            justification = {
                "segment_name": title,
                "behavioral_evidence": f"Steady baseline behavior: Recency {rec:.0f}d, Frequency {freq:.1f}, AOV ${aov:.2f}.",
                "marketing_objective": "Deepen category adoption and elevate lifetime value through cross-sell.",
                "recommended_action": "Introduce double loyalty points on cross-category purchases.",
                "expected_direction_of_impact": "Increase multi-category customer proportion by 20%.",
                "kpi_to_monitor": "Category diversity index, repeat purchase rate, loyalty points redemption.",
                "validation_method": "Points multiplier promotion vs standard catalog newsletter.",
                "confidence_level": "Moderate (Standard core behavioral metrics)."
            }

        strategies[code] = {
            "segment_title": title,
            "segment_size": size,
            "lifecycle_stage": lifecycle_stage,
            "campaign_strategy": {
                "objective": objective,
                "channel": channel,
                "core_message": core_msg,
                "target_category": product_cat,
                "content_format": content_fmt,
                "call_to_action": cta,
                "incentive": incentive,
                "timing": timing,
                "follow_up": follow_up
            },
            "justification": justification
        }
        
    return strategies
