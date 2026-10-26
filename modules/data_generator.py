"""
SegmentIQ AI - Synthetic Data Generator
Generates realistic, statistically sound e-commerce datasets with embedded behavioral archetypes.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(
    num_customers: int = 1200,
    seed: int = 42,
    output_dir: str = "data"
):
    """
    Generates customers.csv, transactions.csv, interactions.csv, and campaign_experiments.csv
    with controlled behavioral archetypes for clustering & segmentation discovery.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_end_date = datetime(2026, 3, 1)
    base_start_date = datetime(2024, 6, 1)
    
    # Define customer archetypes to embed realistic behavioral variety:
    # 0: High-Value Loyal (18%)
    # 1: Frequent Budget / Everyday (24%)
    # 2: Occasion / Premium Spender (15%)
    # 3: At-Risk / Dormant (20%)
    # 4: Discount Hunter (13%)
    # 5: Engaged Browser / Low Spender (10%)
    archetype_probs = [0.18, 0.24, 0.15, 0.20, 0.13, 0.10]
    archetypes = np.random.choice(range(6), size=num_customers, p=archetype_probs)
    
    channels = ["Organic Search", "Paid Social", "Google Ads", "Referral", "Email Marketing", "Influencer"]
    channel_weights = [0.25, 0.22, 0.20, 0.15, 0.10, 0.08]
    
    locations = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Austin, TX", 
                 "Seattle, WA", "Miami, FL", "Denver, CO", "Atlanta, GA", "Boston, MA"]
    
    genders = ["Female", "Male", "Non-Binary", "Prefer not to say"]
    gender_weights = [0.49, 0.44, 0.04, 0.03]
    
    customer_types = ["Direct-to-Consumer", "Retail Member", "Small Business / Pro"]
    cust_type_weights = [0.70, 0.22, 0.08]
    
    # 1. GENERATE CUSTOMERS
    customer_rows = []
    for i in range(num_customers):
        cid = f"CUST-{i+1:04d}"
        arch = archetypes[i]
        
        # Age distribution
        if arch == 0:  # Loyal high-value often slightly older/established
            age = int(np.clip(np.random.normal(44, 10), 24, 70))
        elif arch == 4 or arch == 5: # Young deal hunters / digital native browsers
            age = int(np.clip(np.random.normal(29, 7), 18, 55))
        else:
            age = int(np.clip(np.random.normal(36, 11), 18, 72))
            
        gender = np.random.choice(genders, p=gender_weights)
        location = np.random.choice(locations)
        
        # Registration date
        if arch == 3:  # At-risk registered earlier
            days_back = np.random.randint(350, 600)
        elif arch == 5:  # Browsers registered more recently
            days_back = np.random.randint(30, 250)
        else:
            days_back = np.random.randint(45, 580)
            
        reg_date = (base_end_date - timedelta(days=days_back)).strftime("%Y-%m-%d")
        channel = np.random.choice(channels, p=channel_weights)
        c_type = np.random.choice(customer_types, p=cust_type_weights)
        
        customer_rows.append({
            "customer_id": cid,
            "age": age,
            "gender": gender,
            "location": location,
            "registration_date": reg_date,
            "acquisition_channel": channel,
            "customer_type": c_type,
            "_archetype": arch  # internal label for generating coherent transactions, stripped on export if needed
        })
        
    df_customers = pd.DataFrame(customer_rows)
    
    # 2. GENERATE TRANSACTIONS
    categories = ["Electronics", "Apparel & Fashion", "Home & Living", "Health & Beauty", "Sports & Outdoors", "Books & Media"]
    cat_prices = {
        "Electronics": (80, 550),
        "Apparel & Fashion": (25, 160),
        "Home & Living": (35, 220),
        "Health & Beauty": (15, 85),
        "Sports & Outdoors": (30, 180),
        "Books & Media": (12, 45)
    }
    payment_methods = ["Credit Card", "PayPal", "Apple Pay", "Debit Card", "Buy Now Pay Later"]
    
    transaction_rows = []
    tx_counter = 10001
    
    for idx, cust in df_customers.iterrows():
        cid = cust["customer_id"]
        arch = cust["_archetype"]
        reg_dt = datetime.strptime(cust["registration_date"], "%Y-%m-%d")
        
        # Transaction count per archetype
        if arch == 0:  # High-Value Loyal
            n_tx = np.random.randint(7, 16)
            recency_max_days = 30
            discount_range = (0.0, 0.08)
            cat_pref = ["Electronics", "Apparel & Fashion", "Home & Living"]
        elif arch == 1:  # Budget Everyday Frequent
            n_tx = np.random.randint(9, 20)
            recency_max_days = 25
            discount_range = (0.05, 0.20)
            cat_pref = ["Health & Beauty", "Books & Media", "Apparel & Fashion"]
        elif arch == 2:  # Occasion Premium
            n_tx = np.random.randint(1, 4)
            recency_max_days = 75
            discount_range = (0.0, 0.05)
            cat_pref = ["Electronics", "Home & Living"]
        elif arch == 3:  # At-Risk / Dormant
            n_tx = np.random.randint(2, 6)
            recency_max_days = 300 # Haven't bought in 120-300 days
            discount_range = (0.0, 0.15)
            cat_pref = ["Apparel & Fashion", "Home & Living", "Sports & Outdoors"]
        elif arch == 4:  # Discount Hunter
            n_tx = np.random.randint(4, 9)
            recency_max_days = 60
            discount_range = (0.20, 0.45)
            cat_pref = ["Apparel & Fashion", "Sports & Outdoors", "Health & Beauty"]
        else:  # Engaged Browser / Low Spender
            n_tx = np.random.choice([0, 1], p=[0.45, 0.55])
            recency_max_days = 90
            discount_range = (0.10, 0.30)
            cat_pref = ["Books & Media", "Health & Beauty"]
            
        if n_tx == 0:
            continue
            
        # Generate transaction dates
        tx_dates = []
        if arch == 3:  # At-Risk: all transactions stopped 120 to 300 days ago
            last_date = base_end_date - timedelta(days=np.random.randint(120, 280))
            active_span = max(10, (last_date - reg_dt).days)
            for _ in range(n_tx):
                t_dt = reg_dt + timedelta(days=random.randint(1, active_span))
                tx_dates.append(t_dt)
        else:
            last_date = base_end_date - timedelta(days=np.random.randint(1, recency_max_days))
            active_span = max(15, (last_date - reg_dt).days)
            for _ in range(n_tx):
                t_dt = reg_dt + timedelta(days=random.randint(1, active_span))
                tx_dates.append(t_dt)
                
        tx_dates.sort()
        
        for t_dt in tx_dates:
            tx_id = f"TX-{tx_counter}"
            tx_counter += 1
            
            # Select category
            if random.random() < 0.75:
                cat = random.choice(cat_pref)
            else:
                cat = random.choice(categories)
                
            min_p, max_p = cat_prices[cat]
            if arch == 0 or arch == 2:  # Premium
                unit_price = round(np.random.uniform(min_p * 1.2, max_p * 1.3), 2)
                qty = np.random.randint(1, 4)
            elif arch == 1:  # Budget
                unit_price = round(np.random.uniform(min_p * 0.7, min_p * 1.3), 2)
                qty = np.random.randint(1, 3)
            else:
                unit_price = round(np.random.uniform(min_p, max_p), 2)
                qty = np.random.randint(1, 3)
                
            disc_pct = round(np.random.uniform(discount_range[0], discount_range[1]), 2)
            gross = round(unit_price * qty, 2)
            disc_amt = round(gross * disc_pct, 2)
            net_amt = round(gross - disc_amt, 2)
            
            # Order status: Completed (92%), Refunded (5%), Cancelled (3%)
            status_draw = random.random()
            if status_draw < 0.92:
                status = "Completed"
            elif status_draw < 0.97:
                status = "Refunded"
            else:
                status = "Cancelled"
                
            pay_method = random.choice(payment_methods)
            prod_id = f"PRD-{cat[:3].upper()}-{np.random.randint(101, 399)}"
            
            transaction_rows.append({
                "transaction_id": tx_id,
                "customer_id": cid,
                "transaction_date": t_dt.strftime("%Y-%m-%d"),
                "product_id": prod_id,
                "product_category": cat,
                "quantity": qty,
                "unit_price": unit_price,
                "discount": disc_amt,
                "transaction_amount": net_amt,
                "payment_method": pay_method,
                "order_status": status
            })
            
    df_transactions = pd.DataFrame(transaction_rows)
    
    # 3. GENERATE INTERACTIONS
    interaction_types = ["Email Open", "Email Click", "Website Visit", "App Session", "Product View", "Support Contact", "Campaign Response"]
    interaction_weights = [0.35, 0.15, 0.22, 0.13, 0.08, 0.03, 0.04]
    
    campaigns = ["CAMP-FALL25", "CAMP-HOLIDAY25", "CAMP-SPRING26", "CAMP-VIP-REWARD", "CAMP-WINBACK", "CAMP-FLASH-DEAL"]
    channels_interact = ["Email", "Mobile App", "Website", "SMS", "Support Portal"]
    
    interaction_rows = []
    inter_counter = 50001
    
    for idx, cust in df_customers.iterrows():
        cid = cust["customer_id"]
        arch = cust["_archetype"]
        reg_dt = datetime.strptime(cust["registration_date"], "%Y-%m-%d")
        
        # Interactions frequency per archetype
        if arch == 0:  # High-Value: high open, high engagement
            n_inter = np.random.randint(18, 38)
            last_inter_days = np.random.randint(1, 15)
        elif arch == 1:  # Budget frequent: moderate-high
            n_inter = np.random.randint(14, 30)
            last_inter_days = np.random.randint(2, 20)
        elif arch == 2:  # Premium occasion: low frequency, selective
            n_inter = np.random.randint(5, 14)
            last_inter_days = np.random.randint(10, 60)
        elif arch == 3:  # At-Risk: low & stopped engaging recently
            n_inter = np.random.randint(3, 10)
            last_inter_days = np.random.randint(70, 220)
        elif arch == 4:  # Discount Hunter: spikes around promo campaigns
            n_inter = np.random.randint(12, 26)
            last_inter_days = np.random.randint(5, 45)
        else:  # Engaged Browser: very high browsing, low purchasing
            n_inter = np.random.randint(25, 55)
            last_inter_days = np.random.randint(1, 10)
            
        inter_dates = []
        end_inter_dt = base_end_date - timedelta(days=last_inter_days)
        time_span = max(10, (end_inter_dt - reg_dt).days)
        
        for _ in range(n_inter):
            i_dt = reg_dt + timedelta(days=random.randint(1, time_span))
            inter_dates.append(i_dt)
            
        inter_dates.sort()
        
        for i_dt in inter_dates:
            inter_id = f"INT-{inter_counter}"
            inter_counter += 1
            
            itype = np.random.choice(interaction_types, p=interaction_weights)
            
            # Engagement value (1 to 10 scale)
            if itype == "Product View":
                eng_val = round(np.random.uniform(4.0, 7.5), 1)
                chan = "Website" if random.random() < 0.6 else "Mobile App"
            elif itype == "Email Click":
                eng_val = round(np.random.uniform(6.0, 9.0), 1)
                chan = "Email"
            elif itype == "Campaign Response":
                eng_val = round(np.random.uniform(8.0, 10.0), 1)
                chan = random.choice(["Email", "SMS", "Mobile App"])
            elif itype == "Support Contact":
                eng_val = round(np.random.uniform(3.0, 6.0), 1)
                chan = "Support Portal"
            elif itype == "App Session":
                eng_val = round(np.random.uniform(5.0, 8.5), 1)
                chan = "Mobile App"
            elif itype == "Website Visit":
                eng_val = round(np.random.uniform(3.0, 6.5), 1)
                chan = "Website"
            else: # Email Open
                eng_val = round(np.random.uniform(1.5, 4.0), 1)
                chan = "Email"
                
            camp = random.choice(campaigns) if random.random() < 0.65 else None
            
            interaction_rows.append({
                "interaction_id": inter_id,
                "customer_id": cid,
                "interaction_date": i_dt.strftime("%Y-%m-%d"),
                "interaction_type": itype,
                "channel": chan,
                "campaign_id": camp,
                "engagement_value": eng_val
            })
            
    df_interactions = pd.DataFrame(interaction_rows)
    
    # 4. GENERATE A/B CAMPAIGN EXPERIMENT DATA
    # Test: Segment-personalized offer vs standard generic discount across customer segments
    exp_rows = []
    for exp_id in range(1, 1501):
        cid = f"CUST-{random.randint(1, num_customers):04d}"
        variant = "Variant_B_Personalized" if random.random() < 0.5 else "Variant_A_Control"
        channel = random.choice(["Email", "SMS", "In-App Notification"])
        
        # Base conversion rate: Control ~ 5.5%, Variant B ~ 9.8% (statistically significant lift)
        if variant == "Variant_B_Personalized":
            p_conv = 0.098
            val_mean = 145.0
        else:
            p_conv = 0.054
            val_mean = 115.0
            
        conv = 1 if random.random() < p_conv else 0
        order_val = round(np.random.normal(val_mean, 35), 2) if conv == 1 else 0.0
        if order_val < 0:
            order_val = 25.0
            
        exp_rows.append({
            "experiment_id": f"EXP-{exp_id:04d}",
            "customer_id": cid,
            "campaign_name": "Q1 Retention & Cross-Sell Lift Test",
            "variant": variant,
            "channel": channel,
            "converted": conv,
            "order_value": order_val,
            "timestamp": (base_end_date - timedelta(days=random.randint(5, 40))).strftime("%Y-%m-%d")
        })
        
    df_experiments = pd.DataFrame(exp_rows)
    
    # Save files to output directory
    df_customers_clean = df_customers.drop(columns=["_archetype"])
    customers_path = os.path.join(output_dir, "customers.csv")
    transactions_path = os.path.join(output_dir, "transactions.csv")
    interactions_path = os.path.join(output_dir, "interactions.csv")
    experiments_path = os.path.join(output_dir, "campaign_experiments.csv")
    
    df_customers_clean.to_csv(customers_path, index=False)
    df_transactions.to_csv(transactions_path, index=False)
    df_interactions.to_csv(interactions_path, index=False)
    df_experiments.to_csv(experiments_path, index=False)
    
    print(f"Generated {len(df_customers_clean)} customers -> {customers_path}")
    print(f"Generated {len(df_transactions)} transactions -> {transactions_path}")
    print(f"Generated {len(df_interactions)} interactions -> {interactions_path}")
    print(f"Generated {len(df_experiments)} campaign experiment rows -> {experiments_path}")
    
    return {
        "customers": df_customers_clean,
        "transactions": df_transactions,
        "interactions": df_interactions,
        "experiments": df_experiments
    }

if __name__ == "__main__":
    generate_synthetic_data()
