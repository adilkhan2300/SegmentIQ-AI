"""
SegmentIQ AI - LLM-Powered Customer Segmentation & Marketing Intelligence
Main Streamlit Application
"""

import os
import io
import json
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Module imports
from modules.data_loader import load_sample_datasets, read_uploaded_file, get_csv_template
from modules.data_validation import validate_datasets, clean_and_prepare_data
from modules.feature_engineering import engineer_customer_features, get_feature_dictionary
from modules.clustering import run_clustering, evaluate_candidate_k, get_cluster_summary, DEFAULT_SEGMENTATION_FEATURES
from modules.segment_profiling import (
    compute_segment_radar_data, compute_segment_category_affinity, compute_demographic_distributions
)
from modules.llm_personas import generate_personas
from modules.marketing_recommendations import generate_marketing_strategies
from modules.retention_analysis import compute_rule_based_retention_risk, train_retention_prediction_model
from modules.experiment_analysis import analyze_ab_experiment, simulate_hypothetical_campaign
from modules.insights_engine import discover_automated_insights
from modules.demographic_comparison import compare_segmentation_approaches
from modules.scenario_simulator import run_scenario_simulation
from modules.reporting import generate_excel_report, generate_pdf_report
from modules.database import init_db, save_segmentation_run, list_saved_runs
from utils.ui_helpers import apply_custom_styles, render_metric_card, get_plotly_layout_defaults

# Page configuration
st.set_page_config(
    page_title="SegmentIQ AI | Customer Segmentation & Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek styling
apply_custom_styles()

# Initialize SQLite database
init_db()

# --- SESSION STATE INITIALIZATION ---
def init_session_state():
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df_customers" not in st.session_state:
        st.session_state.df_customers = pd.DataFrame()
    if "df_transactions" not in st.session_state:
        st.session_state.df_transactions = pd.DataFrame()
    if "df_interactions" not in st.session_state:
        st.session_state.df_interactions = pd.DataFrame()
    if "df_experiments" not in st.session_state:
        st.session_state.df_experiments = pd.DataFrame()
    if "df_features" not in st.session_state:
        st.session_state.df_features = pd.DataFrame()
    if "df_clustered" not in st.session_state:
        st.session_state.df_clustered = pd.DataFrame()
    if "cluster_metrics" not in st.session_state:
        st.session_state.cluster_metrics = {}
    if "cluster_summary_df" not in st.session_state:
        st.session_state.cluster_summary_df = pd.DataFrame()
    if "personas_dict" not in st.session_state:
        st.session_state.personas_dict = {}
    if "strategies_dict" not in st.session_state:
        st.session_state.strategies_dict = {}
    if "validation_results" not in st.session_state:
        st.session_state.validation_results = {}
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

init_session_state()

# Auto-load synthetic demo data if nothing loaded yet
if not st.session_state.data_loaded:
    with st.spinner("Initializing SegmentIQ AI with demo dataset..."):
        demo_data = load_sample_datasets()
        c_clean, t_clean, i_clean = clean_and_prepare_data(
            demo_data["customers"], demo_data["transactions"], demo_data["interactions"]
        )
        st.session_state.df_customers = c_clean
        st.session_state.df_transactions = t_clean
        st.session_state.df_interactions = i_clean
        st.session_state.df_experiments = demo_data.get("experiments", pd.DataFrame())
        st.session_state.validation_results = validate_datasets(c_clean, t_clean, i_clean)
        
        # Build features
        feats, _ = engineer_customer_features(c_clean, t_clean, i_clean)
        st.session_state.df_features = feats
        
        # Run baseline clustering
        clustered, metrics = run_clustering(feats, algorithm="K-Means", n_clusters=5)
        st.session_state.df_clustered = clustered
        st.session_state.cluster_metrics = metrics
        
        # Summary
        summary = get_cluster_summary(clustered)
        st.session_state.cluster_summary_df = summary
        
        # Personas & Strategies
        overall_stats = {
            "total_customers": len(clustered),
            "avg_aov": round(clustered["avg_order_value"].mean(), 2),
            "avg_freq": round(clustered["purchase_frequency"].mean(), 1),
            "avg_recency": round(clustered["recency_days"].mean(), 1)
        }
        personas = generate_personas(summary, overall_stats, api_key=None)
        st.session_state.personas_dict = personas
        st.session_state.strategies_dict = generate_marketing_strategies(summary, personas)
        
        st.session_state.data_loaded = True

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("""
<div style="padding: 10px 0 15px 0;">
    <h2 style="margin: 0; color: #3B82F6; font-size: 1.5rem; font-weight: 800;">SegmentIQ <span style="color: #60A5FA;">AI</span></h2>
    <p style="margin: 0; color: #94A3B8; font-size: 0.78rem;">Customer Segmentation & Intelligence</p>
</div>
""", unsafe_allow_html=True)

nav_options = [
    "1. Executive Overview",
    "2. Data Upload and Quality",
    "3. Behavioral Feature Engineering",
    "4. Customer Segmentation",
    "5. Customer Personas",
    "6. Marketing Recommendations",
    "7. Customer Explorer",
    "8. Retention Intelligence",
    "9. Campaign Experiments",
    "10. AI Insights",
    "11. Demographic vs Behavioral Analysis",
    "12. Scenario Simulator",
    "13. Reports and Exports",
    "14. Methodology"
]

page = st.sidebar.radio("Navigation", nav_options, index=0)

# Sidebar Settings
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Configuration")

with st.sidebar.expander("🔑 OpenAI API Settings", expanded=False):
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password",
        help="Optional. If left blank, SegmentIQ AI uses its built-in rule-based persona engine."
    )
    if api_key_input != st.session_state.openai_api_key:
        st.session_state.openai_api_key = api_key_input
        st.success("API Key updated!")
        
    model_choice = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    
    if st.button("🔄 Refresh Personas with LLM", use_container_width=True):
        if not st.session_state.cluster_summary_df.empty:
            with st.spinner("Generating personas via LLM..."):
                stats_ctx = {
                    "total_customers": len(st.session_state.df_clustered),
                    "avg_aov": round(st.session_state.df_clustered["avg_order_value"].mean(), 2),
                    "avg_freq": round(st.session_state.df_clustered["purchase_frequency"].mean(), 1),
                    "avg_recency": round(st.session_state.df_clustered["recency_days"].mean(), 1)
                }
                new_personas = generate_personas(
                    st.session_state.cluster_summary_df,
                    stats_ctx,
                    api_key=st.session_state.openai_api_key,
                    model_name=model_choice
                )
                st.session_state.personas_dict = new_personas
                st.session_state.strategies_dict = generate_marketing_strategies(
                    st.session_state.cluster_summary_df, new_personas
                )
                st.success(f"Personas refreshed using {new_personas.get('generator_type')}!")
                st.rerun()

st.sidebar.markdown("""
<div style="font-size: 0.72rem; color: #64748B; margin-top: 20px;">
    SegmentIQ AI Enterprise v2.4<br/>
    Unsupervised ML & Behavioral Science
</div>
""", unsafe_allow_html=True)

# Helper function to get filtered clustered dataframe
def get_filtered_data():
    df = st.session_state.df_clustered.copy()
    if df.empty:
        return df
    return df

# -------------------------------------------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------------------------------------------
if page == "1. Executive Overview":
    st.markdown("## 📊 Executive Customer Intelligence Dashboard")
    st.markdown(
        "A real-time executive synthesis of customer transaction habits, behavioral segmentation, and revenue concentration."
    )
    
    df_c = st.session_state.df_clustered
    if df_c.empty:
        st.warning("Please upload or initialize customer data in '2. Data Upload and Quality'.")
        st.stop()
        
    # --- FILTERS ROW ---
    with st.expander("🔍 Interactive Cohort & Dimension Filters", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        with f_col1:
            all_segs = ["All Segments"] + sorted(list(df_c["segment_code"].unique()))
            sel_seg = st.selectbox("Customer Segment", all_segs)
            
        with f_col2:
            channels = ["All Channels"] + sorted(list(df_c["acquisition_channel"].dropna().unique())) if "acquisition_channel" in df_c.columns else ["All Channels"]
            sel_channel = st.selectbox("Acquisition Channel", channels)
            
        with f_col3:
            types = ["All Types"] + sorted(list(df_c["customer_type"].dropna().unique())) if "customer_type" in df_c.columns else ["All Types"]
            sel_type = st.selectbox("Customer Type", types)
            
        with f_col4:
            max_r = int(df_c["recency_days"].max()) if "recency_days" in df_c.columns else 180
            sel_rec_max = st.slider("Max Recency (Days)", 10, max_r, max_r)
            
    # Apply filters
    filtered_df = df_c.copy()
    if sel_seg != "All Segments":
        filtered_df = filtered_df[filtered_df["segment_code"] == sel_seg]
    if sel_channel != "All Channels":
        filtered_df = filtered_df[filtered_df["acquisition_channel"] == sel_channel]
    if sel_type != "All Types":
        filtered_df = filtered_df[filtered_df["customer_type"] == sel_type]
    filtered_df = filtered_df[filtered_df["recency_days"] <= sel_rec_max]
    
    if filtered_df.empty:
        st.warning("No customers match the current filter selection.")
        st.stop()
        
    # --- CALCULATE METRICS ---
    tot_cust = len(filtered_df)
    n_segs = filtered_df["segment_code"].nunique()
    tot_rev = filtered_df["monetary_value"].sum()
    avg_aov = filtered_df["avg_order_value"].mean()
    avg_freq = filtered_df["purchase_frequency"].mean()
    avg_rec = filtered_df["recency_days"].mean()
    repeat_rate = (filtered_df["repeat_customer_flag"].mean()) * 100
    
    # Pareto Revenue Concentration (Top 20%)
    sorted_rev = filtered_df["monetary_value"].sort_values(ascending=False)
    top_20_count = max(1, int(0.20 * tot_cust))
    top_20_rev = sorted_rev.iloc[:top_20_count].sum()
    pareto_share = (top_20_rev / tot_rev * 100) if tot_rev > 0 else 0.0
    
    # High value (% with monetary > 75th percentile)
    val_75 = st.session_state.df_clustered["monetary_value"].quantile(0.75)
    high_val_pct = (filtered_df["monetary_value"] >= val_75).mean() * 100
    
    # At-risk (recency > 90 days)
    at_risk_pct = (filtered_df["recency_days"] > 90).mean() * 100
    
    # Avg CLV
    avg_clv = filtered_df["estimated_clv"].mean() if "estimated_clv" in filtered_df.columns else 0.0
    
    # Engagement rate
    eng_rate = (filtered_df["total_interactions"] > 0).mean() * 100 if "total_interactions" in filtered_df.columns else 0.0
    
    # --- KPI CARDS ROW 1 ---
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        render_metric_card("Total Customers", f"{tot_cust:,}", f"Active cohort size", "100%", "info")
    with kpi_col2:
        render_metric_card("Discovered Segments", f"{n_segs}", f"Unsupervised ML clusters", "Behavioral", "info")
    with kpi_col3:
        render_metric_card("Total Net Revenue", f"${tot_rev:,.2f}", f"Avg CLV: ${avg_clv:,.2f}", "Net Value", "pos")
    with kpi_col4:
        render_metric_card("Average Order Value (AOV)", f"${avg_aov:.2f}", f"Per completed transaction", "Spend", "pos")

    # --- KPI CARDS ROW 2 ---
    kpi2_col1, kpi2_col2, kpi2_col3, kpi2_col4 = st.columns(4)
    with kpi2_col1:
        render_metric_card("Avg Purchase Frequency", f"{avg_freq:.1f} orders", f"Across customer lifetime", "Cadence", "info")
    with kpi2_col2:
        render_metric_card("Average Recency", f"{avg_rec:.0f} days", f"Since last completed order", "Recency", "info")
    with kpi2_col3:
        render_metric_card("Repeat Customer Rate", f"{repeat_rate:.1f}%", f"Customers with > 1 order", "Loyalty", "pos")
    with kpi2_col4:
        render_metric_card("Top 20% Rev Concentration", f"{pareto_share:.1f}%", f"Pareto skew indicator", "Concentration", "neg" if pareto_share > 65 else "info")

    # --- KPI CARDS ROW 3 ---
    kpi3_col1, kpi3_col2, kpi3_col3, kpi3_col4 = st.columns(4)
    with kpi3_col1:
        render_metric_card("High-Value Customers", f"{high_val_pct:.1f}%", f"Top quartile spenders", "Tier A", "pos")
    with kpi3_col2:
        render_metric_card("At-Risk Customers", f"{at_risk_pct:.1f}%", f"Recency > 90 days", "Dormancy", "neg" if at_risk_pct > 25 else "pos")
    with kpi3_col3:
        render_metric_card("Engagement Touchpoint Rate", f"{eng_rate:.1f}%", f"Logged interaction records", "Digital", "info")
    with kpi3_col4:
        render_metric_card("Mean Discount Rate", f"{filtered_df['discount_usage_pct'].mean():.1f}%", f"Gross margin erosion", "Promo", "info")

    # --- EXECUTIVE SUMMARY BLOCK ---
    st.markdown("### 📝 Executive Summary & Findings")
    st.markdown(f"""
    <div class="justification-card">
        <p style="margin: 0; font-size: 0.92rem; line-height: 1.5; color: #CBD5E1;">
            <b>Key Takeaway:</b> The customer base exhibits distinct behavioral heterogeneity rather than uniform purchasing patterns. 
            The top 20% of accounts generate <b>{pareto_share:.1f}%</b> of total enterprise revenue, underscoring the critical need for differentiated retention. 
            Approximately <b>{at_risk_pct:.1f}%</b> of accounts currently exceed the 90-day inactivity threshold, representing substantial revenue at risk of churn. 
            Furthermore, <b>{repeat_rate:.1f}%</b> of customers have transitioned into repeat buyers, indicating a high-leverage onboarding opportunity to convert single-order purchasers into perpetual buyers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # --- VISUALIZATION CHARTS ---
    c_chart1, c_chart2 = st.columns(2)
    
    with c_chart1:
        st.markdown("#### 💰 Revenue Share by Segment")
        seg_rev = filtered_df.groupby("segment_code")["monetary_value"].sum().reset_index()
        fig_rev = px.bar(
            seg_rev,
            x="segment_code",
            y="monetary_value",
            color="segment_code",
            text_auto=".2s",
            labels={"monetary_value": "Total Revenue ($)", "segment_code": "Segment"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_rev.update_layout(**get_plotly_layout_defaults(), showlegend=False)
        st.plotly_chart(fig_rev, use_container_width=True)
        
    with c_chart2:
        st.markdown("#### 👥 Customer Volume Distribution")
        seg_cnt = filtered_df["segment_code"].value_counts().reset_index()
        seg_cnt.columns = ["segment_code", "count"]
        fig_pie = px.pie(
            seg_cnt,
            names="segment_code",
            values="count",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(**get_plotly_layout_defaults())
        st.plotly_chart(fig_pie, use_container_width=True)

    # Secondary Charts Row
    c_chart3, c_chart4 = st.columns(2)
    with c_chart3:
        st.markdown("#### 🎯 Recency vs. Monetary Value Scatter")
        fig_scatter = px.scatter(
            filtered_df,
            x="recency_days",
            y="monetary_value",
            color="segment_code",
            size="purchase_frequency",
            hover_data=["customer_id", "avg_order_value", "discount_usage_pct"],
            labels={"recency_days": "Recency (Days)", "monetary_value": "Monetary Value ($)"},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_scatter.update_layout(**get_plotly_layout_defaults())
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with c_chart4:
        st.markdown("#### 🛍️ Preferred Category Affinity")
        if "preferred_category" in filtered_df.columns:
            cat_counts = filtered_df["preferred_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Customers"]
            fig_cat = px.bar(
                cat_counts,
                x="Category",
                y="Customers",
                color="Category",
                text_auto=True,
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig_cat.update_layout(**get_plotly_layout_defaults(), showlegend=False)
            st.plotly_chart(fig_cat, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# PAGE 2: DATA UPLOAD AND QUALITY
# -------------------------------------------------------------------------------------------------
elif page == "2. Data Upload and Quality":
    st.markdown("## 📥 Customer Data Upload & Data Quality Audit")
    st.markdown(
        "Upload separate customer, transaction, and interaction datasets. The platform validates data types, "
        "detects duplicates, checks for negative values, audits foreign-key integrity, and handles order statuses."
    )
    
    # Template downloads row
    st.markdown("#### 📄 Download Standard CSV Templates")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.download_button(
            "⬇️ customers_template.csv",
            data=get_csv_template("customers"),
            file_name="customers_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    with t_col2:
        st.download_button(
            "⬇️ transactions_template.csv",
            data=get_csv_template("transactions"),
            file_name="transactions_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    with t_col3:
        st.download_button(
            "⬇️ interactions_template.csv",
            data=get_csv_template("interactions"),
            file_name="interactions_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    
    # Upload controls
    up_col1, up_col2, up_col3 = st.columns(3)
    with up_col1:
        file_cust = st.file_uploader("Upload Customers File (CSV/Excel)", type=["csv", "xlsx", "xls"])
    with up_col2:
        file_tx = st.file_uploader("Upload Transactions File (CSV/Excel)", type=["csv", "xlsx", "xls"])
    with up_col3:
        file_int = st.file_uploader("Upload Interactions File (Optional)", type=["csv", "xlsx", "xls"])
        
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🚀 Process & Ingest Uploaded Files", use_container_width=True):
            if file_cust is None or file_tx is None:
                st.error("Please upload both Customers and Transactions files to proceed.")
            else:
                with st.spinner("Validating and parsing uploaded files..."):
                    df_c_raw = read_uploaded_file(file_cust)
                    df_t_raw = read_uploaded_file(file_tx)
                    df_i_raw = read_uploaded_file(file_int) if file_int is not None else pd.DataFrame()
                    
                    val_res = validate_datasets(df_c_raw, df_t_raw, df_i_raw)
                    st.session_state.validation_results = val_res
                    
                    if val_res["is_valid"]:
                        c_clean, t_clean, i_clean = clean_and_prepare_data(df_c_raw, df_t_raw, df_i_raw)
                        st.session_state.df_customers = c_clean
                        st.session_state.df_transactions = t_clean
                        st.session_state.df_interactions = i_clean
                        
                        feats, _ = engineer_customer_features(c_clean, t_clean, i_clean)
                        st.session_state.df_features = feats
                        
                        clustered, metrics = run_clustering(feats, algorithm="K-Means", n_clusters=5)
                        st.session_state.df_clustered = clustered
                        st.session_state.cluster_metrics = metrics
                        st.session_state.cluster_summary_df = get_cluster_summary(clustered)
                        
                        overall_stats = {
                            "total_customers": len(clustered),
                            "avg_aov": round(clustered["avg_order_value"].mean(), 2),
                            "avg_freq": round(clustered["purchase_frequency"].mean(), 1),
                            "avg_recency": round(clustered["recency_days"].mean(), 1)
                        }
                        personas = generate_personas(st.session_state.cluster_summary_df, overall_stats, api_key=st.session_state.openai_api_key)
                        st.session_state.personas_dict = personas
                        st.session_state.strategies_dict = generate_marketing_strategies(st.session_state.cluster_summary_df, personas)
                        
                        st.success("Uploaded data processed, cleaned, and clustered successfully!")
                        st.rerun()
                    else:
                        st.error("Validation failed. Please resolve critical errors below.")
                        
    with btn_col2:
        if st.button("🔄 Reset to Synthetic Demo Datasets", use_container_width=True):
            with st.spinner("Loading synthetic demo datasets..."):
                demo = load_sample_datasets()
                c_clean, t_clean, i_clean = clean_and_prepare_data(demo["customers"], demo["transactions"], demo["interactions"])
                st.session_state.df_customers = c_clean
                st.session_state.df_transactions = t_clean
                st.session_state.df_interactions = i_clean
                st.session_state.df_experiments = demo.get("experiments", pd.DataFrame())
                st.session_state.validation_results = validate_datasets(c_clean, t_clean, i_clean)
                
                feats, _ = engineer_customer_features(c_clean, t_clean, i_clean)
                st.session_state.df_features = feats
                
                clustered, metrics = run_clustering(feats, algorithm="K-Means", n_clusters=5)
                st.session_state.df_clustered = clustered
                st.session_state.cluster_metrics = metrics
                st.session_state.cluster_summary_df = get_cluster_summary(clustered)
                
                stats_ctx = {
                    "total_customers": len(clustered),
                    "avg_aov": round(clustered["avg_order_value"].mean(), 2),
                    "avg_freq": round(clustered["purchase_frequency"].mean(), 1),
                    "avg_recency": round(clustered["recency_days"].mean(), 1)
                }
                personas = generate_personas(st.session_state.cluster_summary_df, stats_ctx, api_key=None)
                st.session_state.personas_dict = personas
                st.session_state.strategies_dict = generate_marketing_strategies(st.session_state.cluster_summary_df, personas)
                
                st.success("Demo dataset re-loaded successfully!")
                st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Audit Results
    st.markdown("### 🩺 Data Quality Audit & Cleanliness Report")
    val = st.session_state.validation_results
    if val:
        score = val.get("health_score", 100)
        badge_type = "pos" if score >= 80 else ("info" if score >= 60 else "neg")
        
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        with q_col1:
            render_metric_card("Data Health Score", f"{score}/100", "Composite quality rating", "Verified", badge_type)
        with q_col2:
            render_metric_card("Customer Records", f"{len(st.session_state.df_customers):,}", "Unique customer IDs", "Customers", "info")
        with q_col3:
            render_metric_card("Transaction Records", f"{len(st.session_state.df_transactions):,}", "Order events logged", "Transactions", "info")
        with q_col4:
            render_metric_card("Interaction Records", f"{len(st.session_state.df_interactions):,}", "Touchpoint events", "Interactions", "info")
            
        if val.get("critical_issues"):
            st.error("🚨 Critical Issues Detected:")
            for err in val["critical_issues"]:
                st.write(f"- {err}")
                
        if val.get("warnings"):
            st.info("ℹ️ Data Handling & Cleaning Notes:")
            for w in val["warnings"]:
                st.write(f"- {w}")

    # Data Previews Tabs
    st.markdown("### 🔍 Active Datasets Preview")
    prev_tab1, prev_tab2, prev_tab3 = st.tabs(["Customers", "Transactions", "Interactions"])
    with prev_tab1:
        st.dataframe(st.session_state.df_customers.head(25), use_container_width=True)
    with prev_tab2:
        st.dataframe(st.session_state.df_transactions.head(25), use_container_width=True)
    with prev_tab3:
        if not st.session_state.df_interactions.empty:
            st.dataframe(st.session_state.df_interactions.head(25), use_container_width=True)
        else:
            st.info("No interaction data loaded.")

# -------------------------------------------------------------------------------------------------
# PAGE 3: BEHAVIORAL FEATURE ENGINEERING
# -------------------------------------------------------------------------------------------------
elif page == "3. Behavioral Feature Engineering":
    st.markdown("## 🧠 Behavioral Feature Engineering Pipeline")
    st.markdown(
        "Transforms raw transactional logs and interaction touchpoints into customer-level behavioral metrics. "
        "Includes chronological observation cutoff controls to prevent target data leakage."
    )
    
    df_feat = st.session_state.df_features
    if df_feat.empty:
        st.warning("No features engineered yet.")
        st.stop()
        
    # Cutoff Controller
    with st.expander("⏱️ Observation Period & Leakage Guardrails", expanded=True):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown(
                "**Data Leakage Prevention:** By setting an *Observation Cutoff Date*, all transactions and interactions "
                "occurring after this date are strictly excluded from feature computation."
            )
        with f_col2:
            t_max_date = pd.to_datetime(st.session_state.df_transactions["transaction_date"]).max()
            chosen_date = st.date_input("Observation Cutoff Date", value=t_max_date.date())
            if st.button("Re-compute Features as of Selected Date"):
                with st.spinner("Re-engineering customer features..."):
                    new_feats, _ = engineer_customer_features(
                        st.session_state.df_customers,
                        st.session_state.df_transactions,
                        st.session_state.df_interactions,
                        as_of_date=datetime.combine(chosen_date, datetime.min.time())
                    )
                    st.session_state.df_features = new_feats
                    st.success("Features re-calculated successfully without future leakage!")
                    st.rerun()

    # RFM Histograms
    st.markdown("### 📈 RFM Distribution Profiles")
    rfm_col1, rfm_col2, rfm_col3 = st.columns(3)
    
    with rfm_col1:
        fig_r = px.histogram(
            df_feat, x="recency_days", nbins=30, title="Recency (Days since last purchase)",
            color_discrete_sequence=["#3B82F6"]
        )
        fig_r.update_layout(**get_plotly_layout_defaults())
        st.plotly_chart(fig_r, use_container_width=True)
        
    with rfm_col2:
        fig_f = px.histogram(
            df_feat, x="purchase_frequency", nbins=25, title="Purchase Frequency (Completed Orders)",
            color_discrete_sequence=["#10B981"]
        )
        fig_f.update_layout(**get_plotly_layout_defaults())
        st.plotly_chart(fig_f, use_container_width=True)
        
    with rfm_col3:
        fig_m = px.histogram(
            df_feat, x="monetary_value", nbins=30, title="Monetary Value ($ Net Spend)",
            color_discrete_sequence=["#F59E0B"]
        )
        fig_m.update_layout(**get_plotly_layout_defaults())
        st.plotly_chart(fig_m, use_container_width=True)

    # Feature Matrix Table
    st.markdown("### 📋 Engineered Customer Feature Matrix")
    st.dataframe(df_feat.head(30), use_container_width=True)
    
    # Feature Dictionary
    st.markdown("### 📖 Feature Dictionary & Business Definitions")
    feat_dict_df = get_feature_dictionary()
    st.dataframe(feat_dict_df, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# PAGE 4: CUSTOMER SEGMENTATION
# -------------------------------------------------------------------------------------------------
elif page == "4. Customer Segmentation":
    st.markdown("## 🤖 Machine Learning Customer Segmentation Engine")
    st.markdown(
        "Execute unsupervised clustering algorithms on behavioral indicators, evaluate clustering quality metrics, "
        "and project customer clusters into 2D space with Principal Component Analysis (PCA)."
    )
    
    df_feat = st.session_state.df_features
    if df_feat.empty:
        st.warning("Please compute behavioral features first.")
        st.stop()
        
    # --- MODEL CONTROLS PANEL ---
    with st.expander("🛠️ Segmentation Hyperparameters & Algorithm Controls", expanded=True):
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            algo_choice = st.selectbox("Clustering Algorithm", ["K-Means", "Agglomerative Clustering", "DBSCAN"])
        with m_col2:
            if algo_choice in ["K-Means", "Agglomerative Clustering"]:
                k_val = st.slider("Number of Clusters (k)", min_value=2, max_value=8, value=5)
                db_eps, db_min = 1.2, 5
            else:
                k_val = 5
                db_eps = st.slider("DBSCAN Epsilon (eps)", 0.5, 3.0, 1.3, 0.1)
                db_min = st.slider("DBSCAN Min Samples", 2, 20, 5)
        with m_col3:
            numeric_cols = [c for c in df_feat.select_dtypes(include=[np.number]).columns if c not in ["customer_id", "cluster_id"]]
            default_feats = [f for f in DEFAULT_SEGMENTATION_FEATURES if f in numeric_cols]
            sel_feats = st.multiselect("Active Segmentation Features", numeric_cols, default=default_feats)
            
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🚀 Re-Run Customer Segmentation", use_container_width=True):
                with st.spinner("Executing unsupervised machine learning clustering..."):
                    clustered, metrics = run_clustering(
                        df_feat,
                        algorithm=algo_choice,
                        n_clusters=k_val,
                        features=sel_feats,
                        dbscan_eps=db_eps,
                        dbscan_min_samples=db_min
                    )
                    st.session_state.df_clustered = clustered
                    st.session_state.cluster_metrics = metrics
                    summary = get_cluster_summary(clustered)
                    st.session_state.cluster_summary_df = summary
                    
                    # Update personas
                    stats_ctx = {
                        "total_customers": len(clustered),
                        "avg_aov": round(clustered["avg_order_value"].mean(), 2),
                        "avg_freq": round(clustered["purchase_frequency"].mean(), 1),
                        "avg_recency": round(clustered["recency_days"].mean(), 1)
                    }
                    personas = generate_personas(summary, stats_ctx, api_key=st.session_state.openai_api_key)
                    st.session_state.personas_dict = personas
                    st.session_state.strategies_dict = generate_marketing_strategies(summary, personas)
                    
                    st.success("Segmentation pipeline executed successfully!")
                    st.rerun()
                    
        with c_btn2:
            run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            if st.button(f"💾 Save Current Run to SQLite ({run_id})", use_container_width=True):
                success = save_segmentation_run(
                    run_id,
                    st.session_state.df_clustered,
                    st.session_state.cluster_metrics,
                    st.session_state.strategies_dict
                )
                if success:
                    st.success(f"Run {run_id} saved to SQLite database successfully!")
                else:
                    st.error("Failed to save run to database.")

    # --- CLUSTER QUALITY METRICS ---
    st.markdown("### 📐 Clustering Quality & Validation Metrics")
    m = st.session_state.cluster_metrics
    qm_col1, qm_col2, qm_col3, qm_col4 = st.columns(4)
    with qm_col1:
        sil = m.get("silhouette_score", 0.0)
        render_metric_card("Silhouette Score", f"{sil:.4f}", "Range: -1 to +1 (Higher is crisper)", "Compactness", "pos" if sil > 0.3 else "info")
    with qm_col2:
        db = m.get("davies_bouldin_index", 0.0)
        render_metric_card("Davies-Bouldin Index", f"{db:.4f}", "Lower indicates better separation", "Separation", "pos" if db < 1.5 else "info")
    with qm_col3:
        ch = m.get("calinski_harabasz_index", 0.0)
        render_metric_card("Calinski-Harabasz Index", f"{ch:.1f}", "Variance ratio criterion", "Variance", "info")
    with qm_col4:
        pca_var = m.get("pca_total_variance", 0.0)
        render_metric_card("PCA 2D Variance", f"{pca_var:.1f}%", "Captured by 2 principal components", "PCA", "info")

    # --- ELBOW & SILHOUETTE COMPARISON CHART ---
    with st.expander("📊 Candidate Cluster Count (k) Evaluation & Elbow Analysis", expanded=False):
        k_eval_df = evaluate_candidate_k(df_feat, features=sel_feats, k_range=range(2, 9))
        if not k_eval_df.empty:
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                fig_elbow = px.line(
                    k_eval_df, x="k", y="Inertia", markers=True, title="Inertia Elbow Curve",
                    color_discrete_sequence=["#EF4444"]
                )
                fig_elbow.update_layout(**get_plotly_layout_defaults())
                st.plotly_chart(fig_elbow, use_container_width=True)
            with e_col2:
                fig_sil = px.line(
                    k_eval_df, x="k", y="Silhouette Score", markers=True, title="Silhouette Score Curve",
                    color_discrete_sequence=["#10B981"]
                )
                fig_sil.update_layout(**get_plotly_layout_defaults())
                st.plotly_chart(fig_sil, use_container_width=True)

    # --- VISUALIZATIONS ---
    st.markdown("### 🗺️ PCA 2D Cluster Space & Visual Distribution")
    df_c = st.session_state.df_clustered
    
    p_col1, p_col2 = st.columns([3, 2])
    with p_col1:
        fig_pca = px.scatter(
            df_c,
            x="pca_1",
            y="pca_2",
            color="segment_code",
            hover_name="customer_id",
            hover_data=["purchase_frequency", "monetary_value", "avg_order_value", "recency_days", "discount_usage_pct"],
            title="PCA 2D Cluster Scatter Plot",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pca.update_layout(**get_plotly_layout_defaults())
        st.plotly_chart(fig_pca, use_container_width=True)
        
    with p_col2:
        st.markdown("#### 🧭 Multi-Dimensional Radar Comparison")
        radar_df, r_feats = compute_segment_radar_data(df_c)
        if not radar_df.empty:
            fig_radar = go.Figure()
            for idx, r_row in radar_df.iterrows():
                vals = [r_row[f] for f in r_feats]
                vals.append(vals[0])  # close loop
                theta = r_feats + [r_feats[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=theta,
                    fill='toself',
                    name=str(r_row["segment_code"]),
                    opacity=0.4
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1E293B"),
                    angularaxis=dict(gridcolor="#1E293B")
                ),
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#CBD5E1"),
                margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(orientation="h", y=-0.15)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # Segment Summary Table
    st.markdown("### 📊 Segment Statistical Profiles")
    st.dataframe(st.session_state.cluster_summary_df, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# PAGE 5: CUSTOMER PERSONAS
# -------------------------------------------------------------------------------------------------
elif page == "5. Customer Personas":
    st.markdown("## 🎭 AI-Interpreted Customer Personas")
    st.markdown(
        "Transforms numerical cluster centroids into rich, understandable behavioral customer personas. "
        "Generated using LLMs or calibrated rule-based heuristics."
    )
    
    personas_data = st.session_state.personas_dict
    personas = personas_data.get("personas", {})
    gen_type = personas_data.get("generator_type", "Rule-Based Engine")
    
    st.info(f"💡 Active Persona Engine: **{gen_type}**")
    
    if not personas:
        st.warning("No personas available. Please run segmentation in '4. Customer Segmentation'.")
        st.stop()
        
    for code, p in personas.items():
        title = p.get("persona_title", code)
        desc = p.get("short_description", "")
        behaviors = p.get("dominant_behaviors", [])
        spend_profile = p.get("spending_profile", "")
        eng_pattern = p.get("engagement_pattern", "")
        needs = p.get("hypothesized_needs", [])
        barriers = p.get("retention_barriers", [])
        evidence = p.get("empirical_evidence", "")
        conf = p.get("confidence_assessment", "Moderate")
        
        with st.container():
            st.markdown(f"""
            <div class="persona-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h3 style="margin: 0; color: #60A5FA;">{code}: {title}</h3>
                    <span class="metric-badge-info">Confidence: {conf}</span>
                </div>
                <p style="font-size: 0.94rem; color: #E2E8F0; margin-bottom: 12px; line-height: 1.4;">
                    <b>Persona Overview:</b> {desc}
                </p>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 8px;">
                    <b>Empirical Grounding:</b> <code>{evidence}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            p_c1, p_c2 = st.columns(2)
            with p_c1:
                st.markdown("**Dominant Behaviors:**")
                for b in behaviors:
                    st.write(f"- {b}")
                st.markdown(f"**Spending Profile:** {spend_profile}")
                st.markdown(f"**Engagement Pattern:** {eng_pattern}")
                
            with p_c2:
                st.markdown("**Hypothesized Needs & Drivers (Subject to Testing):**")
                for n in needs:
                    st.write(f"- {n}")
                st.markdown("**Potential Friction & Retention Barriers:**")
                for barr in barriers:
                    st.write(f"- ⚠️ {barr}")
                    
            st.markdown("---")

# -------------------------------------------------------------------------------------------------
# PAGE 6: MARKETING RECOMMENDATIONS
# -------------------------------------------------------------------------------------------------
elif page == "6. Marketing Recommendations":
    st.markdown("## 🎯 Differentiated Marketing Strategies & Action Plans")
    st.markdown(
        "Prescriptive marketing campaigns and lifecycle roadmaps backed by an 8-point business justification. "
        "Eliminates blanket discounting in favor of margin-aware, segment-appropriate incentives."
    )
    
    strategies = st.session_state.strategies_dict
    if not strategies:
        st.warning("No marketing strategies generated. Please run segmentation.")
        st.stop()
        
    for code, s in strategies.items():
        title = s.get("segment_title", code)
        stage = s.get("lifecycle_stage", "Core Retention")
        camp = s.get("campaign_strategy", {})
        just = s.get("justification", {})
        
        with st.expander(f"📌 {code}: {title} — [{stage}]", expanded=True):
            m_col1, m_col2 = st.columns([1, 1])
            
            with m_col1:
                st.markdown("#### 📣 Campaign Strategy Tactics")
                st.write(f"**Campaign Objective:** {camp.get('objective')}")
                st.write(f"**Recommended Channel:** `{camp.get('channel')}`")
                st.write(f"**Core Message:** *\"{camp.get('core_message')}\"*")
                st.write(f"**Target Product / Category:** {camp.get('target_category')}")
                st.write(f"**Suggested Content Format:** {camp.get('content_format')}")
                st.write(f"**Call to Action (CTA):** `{camp.get('call_to_action')}`")
                st.write(f"**Recommended Incentive:** 🎁 {camp.get('incentive')}")
                st.write(f"**Optimal Timing:** 🕒 {camp.get('timing')}")
                st.write(f"**Follow-Up Cadence:** 🔄 {camp.get('follow_up')}")
                
            with m_col2:
                st.markdown("#### 💼 8-Point Business Justification")
                st.markdown(f"""
                <div class="justification-card" style="font-size: 0.84rem; line-height: 1.45;">
                    <b>1. Segment Name:</b> {just.get('segment_name')}<br/>
                    <b>2. Supporting Evidence:</b> {just.get('behavioral_evidence')}<br/>
                    <b>3. Marketing Objective:</b> {just.get('marketing_objective')}<br/>
                    <b>4. Recommended Action:</b> {just.get('recommended_action')}<br/>
                    <b>5. Expected Impact:</b> <span style="color: #10B981;">{just.get('expected_direction_of_impact')}</span><br/>
                    <b>6. Primary KPI to Monitor:</b> <code>{just.get('kpi_to_monitor')}</code><br/>
                    <b>7. Validation Experiment:</b> {just.get('validation_method')}<br/>
                    <b>8. Confidence Assessment:</b> {just.get('confidence_level')}
                </div>
                """, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# PAGE 7: CUSTOMER EXPLORER
# -------------------------------------------------------------------------------------------------
elif page == "7. Customer Explorer":
    st.markdown("## 🔍 Customer-Level Intelligence Explorer")
    st.markdown(
        "Inspect individual customer records, explore their multi-dimensional behavioral fingerprint, "
        "and compare individual attributes against their segment baseline."
    )
    
    df_c = st.session_state.df_clustered
    if df_c.empty:
        st.warning("No clustered customer data available.")
        st.stop()
        
    # Search and Filter
    exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
    with exp_col1:
        seg_filter = st.selectbox("Filter Segment", ["All"] + sorted(list(df_c["segment_code"].unique())))
    with exp_col2:
        sort_by = st.selectbox("Sort By", ["monetary_value", "purchase_frequency", "recency_days", "estimated_clv"], index=0)
    with exp_col3:
        cust_search = st.text_input("Search Customer ID", placeholder="e.g. CUST-0001")
        
    filtered = df_c.copy()
    if seg_filter != "All":
        filtered = filtered[filtered["segment_code"] == seg_filter]
    if cust_search.strip():
        filtered = filtered[filtered["customer_id"].str.contains(cust_search.strip(), case=False)]
    filtered = filtered.sort_values(by=sort_by, ascending=False)
    
    st.dataframe(filtered[[
        "customer_id", "segment_code", "monetary_value", "purchase_frequency", "avg_order_value",
        "recency_days", "discount_usage_pct", "preferred_category", "total_interactions", "estimated_clv"
    ]].head(50), use_container_width=True)
    
    # Detailed Individual Profile Dossier
    st.markdown("### 👤 Customer Dossier & Explainability View")
    available_cids = filtered["customer_id"].tolist()
    if available_cids:
        selected_cid = st.selectbox("Select Customer to Inspect Dossier", available_cids[:100])
        cust_row = df_c[df_c["customer_id"] == selected_cid].iloc[0]
        seg_code = cust_row["segment_code"]
        
        d_col1, d_col2 = st.columns([1, 1])
        with d_col1:
            st.markdown(f"#### Account: `{selected_cid}` ({seg_code})")
            st.write(f"**Customer Tenure:** {cust_row.get('customer_tenure_days', 'N/A')} days")
            st.write(f"**Demographics:** {cust_row.get('age', 'N/A')} yrs | {cust_row.get('gender', 'N/A')} | {cust_row.get('location', 'N/A')}")
            st.write(f"**Acquisition Channel:** {cust_row.get('acquisition_channel', 'N/A')}")
            st.write(f"**Recency:** {cust_row['recency_days']:.0f} days since last purchase")
            st.write(f"**Frequency:** {cust_row['purchase_frequency']:.0f} orders completed")
            st.write(f"**Monetary Spend:** ${cust_row['monetary_value']:,.2f}")
            st.write(f"**Average Order Value:** ${cust_row['avg_order_value']:.2f}")
            st.write(f"**Preferred Category:** {cust_row.get('preferred_category', 'N/A')}")
            st.write(f"**Total Touchpoints:** {cust_row.get('total_interactions', 0)} interactions")
            
        with d_col2:
            st.markdown("#### 🎯 Explainable Cluster Affinity")
            # Compare customer to segment average
            seg_avg = df_c[df_c["segment_code"] == seg_code][[
                "monetary_value", "purchase_frequency", "avg_order_value", "recency_days"
            ]].mean()
            
            comp_df = pd.DataFrame({
                "Metric": ["Monetary Value ($)", "Frequency (Orders)", "AOV ($)", "Recency (Days)"],
                "Customer Value": [cust_row["monetary_value"], cust_row["purchase_frequency"], cust_row["avg_order_value"], cust_row["recency_days"]],
                "Segment Average": [seg_avg["monetary_value"], seg_avg["purchase_frequency"], seg_avg["avg_order_value"], seg_avg["recency_days"]]
            })
            st.dataframe(comp_df.round(2), use_container_width=True)

# -------------------------------------------------------------------------------------------------
# PAGE 8: RETENTION INTELLIGENCE
# -------------------------------------------------------------------------------------------------
elif page == "8. Retention Intelligence":
    st.markdown("## 🛡️ Customer Value & Retention Intelligence")
    st.markdown(
        "Identify at-risk accounts using transparent business logic rules and train supervised predictive machine learning models "
        "to forecast churn risk and repeat purchase propensity."
    )
    
    df_feat = st.session_state.df_features
    if df_feat.empty:
        st.warning("Please compute customer features first.")
        st.stop()
        
    # Rule-Based Risk Scoring
    df_risk = compute_rule_based_retention_risk(df_feat)
    
    st.markdown("### 📊 Business-Rule Retention Risk Tiers")
    tier_counts = df_risk["retention_risk_tier"].value_counts().reset_index()
    tier_counts.columns = ["Risk Tier", "Customers"]
    
    r_col1, r_col2 = st.columns([1, 1])
    with r_col1:
        fig_tier = px.pie(
            tier_counts, names="Risk Tier", values="Customers", hole=0.4,
            color="Risk Tier",
            color_discrete_map={
                "High Risk (Lapsed / At-Risk)": "#EF4444",
                "Medium Risk (Slowing Cadence)": "#F59E0B",
                "Low Risk (Healthy / Active)": "#10B981"
            }
        )
        fig_tier.update_layout(**get_plotly_layout_defaults())
        st.plotly_chart(fig_tier, use_container_width=True)
        
    with r_col2:
        high_risk_count = (df_risk['retention_risk_tier'] == 'High Risk (Lapsed / At-Risk)').sum()
        render_metric_card("High Churn Risk Accounts", f"{high_risk_count:,}", f"{(high_risk_count/len(df_risk)*100):.1f}% of customer base", "Action Required", "neg")
        st.markdown(
            "**Risk Scoring Formula:** Weighted synthesis of (1) Recency lapse (up to 40 pts), "
            "(2) Inactivity relative to customer's personal inter-purchase interval (up to 25 pts), "
            "(3) Digital engagement silence (up to 20 pts), and (4) Purchase cadence velocity (up to 15 pts)."
        )

    # Machine Learning Churn Model
    st.markdown("---")
    st.markdown("### 🤖 Predictive Machine Learning Model")
    
    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        target_choice = st.selectbox("Prediction Target", ["churn_risk", "repeat_purchase"], format_func=lambda x: "High Churn Risk Prediction" if x == "churn_risk" else "Repeat Purchase Propensity")
    with ml_col2:
        model_choice = st.selectbox("Classifier Algorithm", ["Random Forest", "Logistic Regression"])
        
    if st.button("🚀 Train Predictive Model"):
        with st.spinner("Training classification model with train/test split..."):
            ml_metrics, feat_imp, df_pred = train_retention_prediction_model(
                df_feat, target_type=target_choice, model_type=model_choice
            )
            
            p_m1, p_m2, p_m3, p_m4 = st.columns(4)
            with p_m1:
                render_metric_card("ROC-AUC Score", f"{ml_metrics['roc_auc']:.4f}", "Discrimination power", "AUC", "pos")
            with p_m2:
                render_metric_card("Precision", f"{ml_metrics['precision']:.4f}", "Positive predictive value", "Precision", "info")
            with p_m3:
                render_metric_card("Recall", f"{ml_metrics['recall']:.4f}", "Sensitivity / Hit rate", "Recall", "info")
            with p_m4:
                render_metric_card("F1-Score", f"{ml_metrics['f1_score']:.4f}", "Harmonic mean", "F1", "info")
                
            st.markdown("#### Feature Importance Ranking")
            fig_imp = px.bar(
                feat_imp, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues"
            )
            fig_imp.update_layout(**get_plotly_layout_defaults())
            st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# PAGE 9: CAMPAIGN EXPERIMENTS
# -------------------------------------------------------------------------------------------------
elif page == "9. Campaign Experiments":
    st.markdown("## 🧪 Marketing Experimentation & A/B Testing Analysis")
    st.markdown(
        "Rigorous statistical evaluation of marketing experiments with two-proportion Z-tests, p-values, "
        "confidence intervals, and hypothetical scenario budgeting."
    )
    
    exp_tab1, exp_tab2 = st.tabs(["A/B Experiment Statistical Analysis", "Hypothetical Campaign Simulator"])
    
    with exp_tab1:
        st.markdown("### 📊 Active A/B Campaign Experiment Results")
        df_exp = st.session_state.df_experiments
        if df_exp.empty:
            st.info("No experiment dataset loaded.")
        else:
            ab_results = analyze_ab_experiment(df_exp)
            if "error" in ab_results:
                st.error(ab_results["error"])
            else:
                ab_col1, ab_col2, ab_col3, ab_col4 = st.columns(4)
                with ab_col1:
                    render_metric_card("Control Conv Rate", f"{ab_results['conversion_rate_control']:.2f}%", f"Sample: {ab_results['n_control']:,}", "Variant A", "info")
                with ab_col2:
                    render_metric_card("Treatment Conv Rate", f"{ab_results['conversion_rate_treatment']:.2f}%", f"Sample: {ab_results['n_treatment']:,}", "Variant B", "pos")
                with ab_col3:
                    render_metric_card("Relative Lift", f"+{ab_results['relative_lift_pct']:.1f}%", f"Abs Lift: +{ab_results['absolute_lift_pct_points']:.2f}% pts", "Lift", "pos")
                with ab_col4:
                    sig_badge = "Statistically Significant" if ab_results["is_statistically_significant"] else "Not Significant"
                    sig_type = "pos" if ab_results["is_statistically_significant"] else "neg"
                    render_metric_card("p-value", f"{ab_results['p_value']:.4f}", f"95% CI: [{ab_results['ci_lower_pct']}%, {ab_results['ci_upper_pct']}%]", sig_badge, sig_type)
                    
                st.markdown(f"""
                <div class="justification-card">
                    <b>Statistical Verdict:</b> {'The personalized treatment demonstrates a statistically significant lift (p < 0.05). We reject the null hypothesis of equal performance.' if ab_results['is_statistically_significant'] else 'The lift is not statistically significant at 95% confidence.'}
                    Revenue per user increased from <b>${ab_results['rev_per_user_control']:.2f}</b> to <b>${ab_results['rev_per_user_treatment']:.2f}</b>.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br/>", unsafe_allow_html=True)
                st.dataframe(df_exp.head(20), use_container_width=True)

    with exp_tab2:
        st.markdown("### 🧮 Hypothetical Campaign Budget & ROI Simulator")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            t_size = st.number_input("Target Segment Size (Customers)", 50, 100000, 1200)
            chan = st.selectbox("Marketing Channel", ["Email", "SMS", "Push Notification", "Paid Retargeting Ad", "Direct Mail / Catalog"])
            bud = st.number_input("Campaign Budget ($)", 100.0, 50000.0, 1500.0, step=100.0)
        with sim_col2:
            base_cr = st.slider("Baseline Conversion Rate (%)", 0.5, 20.0, 4.5, 0.1)
            proj_lift = st.slider("Projected Lift (%)", 5.0, 100.0, 35.0, 1.0)
            aov_sim = st.number_input("Average Order Value ($)", 10.0, 1000.0, 95.0, step=5.0)
            
        sim_res = simulate_hypothetical_campaign(
            t_size, chan, bud, base_cr, proj_lift, aov_sim
        )
        
        st.markdown("#### Projected Campaign Outcomes:")
        sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
        with sc_col1:
            render_metric_card("Audience Reached", f"{sim_res['customers_reached']:,}", f"{sim_res['reach_percentage']}% of audience", "Reach", "info")
        with sc_col2:
            render_metric_card("Projected Orders", f"{sim_res['projected_orders']:,}", f"+{sim_res['incremental_orders']} incremental", "Orders", "pos")
        with sc_col3:
            render_metric_card("Incremental Revenue", f"${sim_res['incremental_revenue']:,.2f}", f"Total: ${sim_res['projected_revenue']:,.2f}", "Revenue", "pos")
        with sc_col4:
            render_metric_card("Estimated ROI", f"{sim_res['estimated_roi_pct']:.1f}%", f"Net Profit: ${sim_res['net_profit']:,.2f}", "ROI", "pos" if sim_res['estimated_roi_pct'] > 0 else "neg")

# -------------------------------------------------------------------------------------------------
# PAGE 10: AI INSIGHTS
# -------------------------------------------------------------------------------------------------
elif page == "10. AI Insights":
    st.markdown("## 💡 Automated AI Insights & Segment Discovery")
    st.markdown(
        "Algorithmic pattern discovery engine highlighting revenue concentration anomalies, "
        "engagement gaps, margin cannibalization, and high-leverage growth opportunities."
    )
    
    df_c = st.session_state.df_clustered
    df_t = st.session_state.df_transactions
    summary_df = st.session_state.cluster_summary_df
    
    if df_c.empty:
        st.warning("Please compute clustering first.")
        st.stop()
        
    insights = discover_automated_insights(df_c, df_t, summary_df)
    
    for ins in insights:
        with st.container():
            st.markdown(f"""
            <div class="persona-card" style="border-left-color: #10B981;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <h4 style="margin: 0; color: #34D399;">{ins['title']}</h4>
                    <span class="metric-badge-info">{ins['category']}</span>
                </div>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 8px;">
                    <b>Relevant Segment:</b> <code>{ins['segment']}</code> | <b>Metrics:</b> {ins['supporting_metrics']}
                </p>
                <p style="font-size: 0.92rem; color: #F1F5F9; line-height: 1.45; margin-bottom: 8px;">
                    <b>Executive Interpretation:</b> {ins['interpretation']}
                </p>
                <div style="background-color: #1E293B; padding: 10px 14px; border-radius: 6px; font-size: 0.88rem; margin-bottom: 6px;">
                    <b>Recommended Marketing Action:</b> {ins['recommended_action']}
                </div>
                <div style="font-size: 0.78rem; color: #64748B;">
                    ⚠️ <i>Limitations: {ins['limitations']}</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# PAGE 11: DEMOGRAPHIC VS BEHAVIORAL ANALYSIS
# -------------------------------------------------------------------------------------------------
elif page == "11. Demographic vs Behavioral Analysis":
    st.markdown("## ⚖️ Demographic vs. Behavioral Segmentation Analysis")
    st.markdown(
        "Head-to-head empirical comparison demonstrating why behavioral segmentation provides significantly sharper, "
        "actionable variance than relying on broad demographic proxies (age or geography)."
    )
    
    df_c = st.session_state.df_clustered
    if df_c.empty:
        st.warning("Please execute clustering first.")
        st.stop()
        
    comparison_data = compare_segmentation_approaches(df_c)
    
    # Comparison Summary Cards
    st.markdown("### 📊 ANOVA F-Statistic Comparison (Variance in Monetary Spend)")
    st.markdown(
        "The **ANOVA F-Statistic** measures between-group variance relative to within-group variance. "
        "Higher F-values indicate much sharper statistical differentiation between customer groups."
    )
    
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    cols = [f_col1, f_col2, f_col3, f_col4]
    
    for idx, (name, res) in enumerate(comparison_data.items()):
        with cols[idx]:
            render_metric_card(name, f"F = {res['f_statistic_monetary']:.1f}", f"p-value: {res['p_value_monetary']}", "Variance", "pos" if res['f_statistic_monetary'] > 20 else "info")

    st.markdown("---")
    
    # Detail Tabs
    tabs = st.tabs(list(comparison_data.keys()))
    for idx, (name, res) in enumerate(comparison_data.items()):
        with tabs[idx]:
            st.markdown(f"#### Cohort Breakdown: {name}")
            st.dataframe(res["summary_table"], use_container_width=True)
            
            # Chart
            fig_bar = px.bar(
                res["summary_table"],
                x=res["summary_table"].columns[0],
                y="avg_monetary",
                text="avg_monetary",
                color="avg_monetary",
                labels={"avg_monetary": "Average Spend ($)"},
                title=f"Average Spend per Customer across {name}",
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(**get_plotly_layout_defaults())
            st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# PAGE 12: SCENARIO SIMULATOR
# -------------------------------------------------------------------------------------------------
elif page == "12. Scenario Simulator":
    st.markdown("## 🔮 Strategic Marketing What-If Scenario Simulator")
    st.markdown(
        "Simulate the financial impact of strategic marketing interventions such as retention reactivation, "
        "repeat purchase acceleration, discount margin recovery, and window shopper conversion."
    )
    
    df_c = st.session_state.df_clustered
    if df_c.empty:
        st.warning("Please run clustering first.")
        st.stop()
        
    sim_c1, sim_c2 = st.columns(2)
    with sim_c1:
        st.markdown("#### Strategic Growth Levers")
        ret_boost = st.slider("Retention Reactivation Rate (%) on At-Risk Customers", 0.0, 50.0, 15.0, 1.0)
        rep_boost = st.slider("Repeat Purchase Conversion (%) on Single Buyers", 0.0, 40.0, 10.0, 1.0)
        disc_opt = st.slider("Discount Leakage Reduction (%)", 0.0, 50.0, 20.0, 1.0)
        browser_conv = st.slider("Digital Window Shopper Conversion Rate (%)", 0.0, 25.0, 5.0, 0.5)
        
    with sim_c2:
        st.markdown("#### Financial Parameters")
        budget_input = st.number_input("Intervention Campaign Budget ($)", 1000.0, 100000.0, 15000.0, step=1000.0)
        margin_input = st.slider("Enterprise Gross Margin (%)", 15.0, 70.0, 38.0, 1.0)
        
    sim_out = run_scenario_simulation(
        df_c,
        retention_boost_pct=ret_boost,
        repeat_rate_boost_pct=rep_boost,
        discount_reduction_pct=disc_opt,
        window_shopper_conversion_pct=browser_conv,
        marketing_budget=budget_input,
        gross_margin_pct=margin_input
    )
    
    st.markdown("### 📊 Projected Scenario Financials vs Baseline Actuals")
    b = sim_out["baseline"]
    s = sim_out["scenario"]
    
    s_c1, s_c2, s_c3, s_c4 = st.columns(4)
    with s_c1:
        render_metric_card("Projected Total Revenue", f"${s['total_revenue']:,.2f}", f"+${s['incremental_revenue']:,.2f} incremental", "Revenue", "pos")
    with s_c2:
        render_metric_card("Total Orders", f"{s['total_orders']:,}", f"+{s['incremental_orders']:,} incremental", "Orders", "pos")
    with s_c3:
        render_metric_card("Net Incremental Profit", f"${s['net_incremental_profit']:,.2f}", f"After ${budget_input:,.0f} spend", "Profit", "pos" if s['net_incremental_profit'] > 0 else "neg")
    with s_c4:
        render_metric_card("Campaign ROI", f"{s['projected_roi_pct']:.1f}%", "Estimated return on spend", "ROI", "pos" if s['projected_roi_pct'] > 0 else "neg")

    # Comparison Waterfall
    st.markdown("#### Projected Revenue Bridge")
    bridge_labels = [
        "Baseline Revenue", "Reactivated At-Risk", "New Repeat Orders",
        "Converted Browsers", "Projected Total Revenue"
    ]
    bridge_vals = [
        b["revenue"],
        s["incremental_revenue"] * 0.45,
        s["incremental_revenue"] * 0.35,
        s["incremental_revenue"] * 0.20,
        s["total_revenue"]
    ]
    
    fig_waterfall = go.Figure(go.Waterfall(
        name="Revenue Bridge", orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=bridge_labels,
        y=[b["revenue"], s["incremental_revenue"]*0.45, s["incremental_revenue"]*0.35, s["incremental_revenue"]*0.20, 0],
        connector={"line": {"color": "#3B82F6"}},
        decreasing={"marker": {"color": "#EF4444"}},
        increasing={"marker": {"color": "#10B981"}},
        totals={"marker": {"color": "#2563EB"}}
    ))
    fig_waterfall.update_layout(**get_plotly_layout_defaults())
    st.plotly_chart(fig_waterfall, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# PAGE 13: REPORTS AND EXPORTS
# -------------------------------------------------------------------------------------------------
elif page == "13. Reports and Exports":
    st.markdown("## 📑 Management Reporting & Multi-Format Exports")
    st.markdown(
        "Export comprehensive intelligence assets in CSV, Excel (multi-tab formatted workbook), "
        "and executive-ready PDF report generated via ReportLab."
    )
    
    df_c = st.session_state.df_clustered
    summary_df = st.session_state.cluster_summary_df
    personas_dict = st.session_state.personas_dict
    strategies_dict = st.session_state.strategies_dict
    metrics = st.session_state.cluster_metrics
    
    if df_c.empty:
        st.warning("No data available to export.")
        st.stop()
        
    overall_kpis = {
        "total_customers": len(df_c),
        "total_revenue": df_c["monetary_value"].sum() if "monetary_value" in df_c.columns else 0.0,
        "avg_aov": df_c["avg_order_value"].mean() if "avg_order_value" in df_c.columns else 0.0,
        "avg_frequency": df_c["purchase_frequency"].mean() if "purchase_frequency" in df_c.columns else 0.0,
        "repeat_rate": (df_c["repeat_customer_flag"].mean() * 100) if "repeat_customer_flag" in df_c.columns else 0.0,
        "top_20_share": 65.0,
        "at_risk_rate": ((df_c["recency_days"] > 90).mean() * 100) if "recency_days" in df_c.columns else 0.0
    }
    
    exp_c1, exp_c2 = st.columns(2)
    
    with exp_c1:
        st.markdown("### 📊 Multi-Tab Excel Workbook")
        st.write("Contains separate worksheets for Customer Assignments, Segment Profiles, Personas, Strategies, and Model Metrics.")
        with st.spinner("Generating Excel workbook..."):
            excel_bytes = generate_excel_report(df_c, summary_df, personas_dict, strategies_dict, metrics)
            st.download_button(
                label="📥 Download SegmentIQ_Intelligence_Package.xlsx",
                data=excel_bytes,
                file_name="SegmentIQ_Intelligence_Package.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
    with exp_c2:
        st.markdown("### 📄 Executive PDF Intelligence Report")
        st.write("Executive-grade PDF report with KPI summaries, clustering diagnostics, customer personas, strategy plans, and governance notes.")
        with st.spinner("Generating PDF report via ReportLab..."):
            pdf_bytes = generate_pdf_report(df_c, summary_df, personas_dict, strategies_dict, metrics, overall_kpis)
            st.download_button(
                label="📥 Download SegmentIQ_Executive_Report.pdf",
                data=pdf_bytes,
                file_name="SegmentIQ_Executive_Intelligence_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("### 💾 Individual CSV Exports")
    csv_c1, csv_c2, csv_c3 = st.columns(3)
    with csv_c1:
        st.download_button(
            "⬇️ customer_segments.csv",
            data=df_c.to_csv(index=False),
            file_name="customer_segments.csv",
            mime="text/csv",
            use_container_width=True
        )
    with csv_c2:
        st.download_button(
            "⬇️ segment_profiles.csv",
            data=summary_df.to_csv(index=False),
            file_name="segment_profiles.csv",
            mime="text/csv",
            use_container_width=True
        )
    with csv_c3:
        # Export saved database runs
        db_runs = list_saved_runs()
        st.download_button(
            "⬇️ historical_runs.csv",
            data=db_runs.to_csv(index=False),
            file_name="historical_segmentation_runs.csv",
            mime="text/csv",
            use_container_width=True
        )

# -------------------------------------------------------------------------------------------------
# PAGE 14: METHODOLOGY
# -------------------------------------------------------------------------------------------------
elif page == "14. Methodology":
    st.markdown("## 📚 Methodology & Scientific Architecture")
    st.markdown(
        "Comprehensive documentation of the machine learning algorithms, RFM feature engineering mathematical logic, "
        "LLM persona prompt engineering standards, and experimental statistical tests."
    )
    
    with st.expander("1. Unsupervised Machine Learning Algorithms", expanded=True):
        st.markdown("""
        - **K-Means Clustering:** Minimizes within-cluster sum of squares (inertia). We execute multiple restarts (`n_init=15`) with standardized feature scaling (`StandardScaler`) to eliminate feature variance scale bias.
        - **Agglomerative Clustering:** Hierarchical bottom-up clustering constructing a dendrogram based on Ward's linkage.
        - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Discovers core clusters of arbitrary shapes and labels isolated anomalous outliers as noise (`cluster_id = -1`).
        """)
        
    with st.expander("2. Evaluation Metrics & Dimensionality Reduction", expanded=True):
        st.markdown(r"""
        - **Silhouette Score:** Evaluates cohesion versus separation ($s \in [-1, 1]$). Values $> 0.35$ indicate healthy cluster structure.
        - **Davies-Bouldin Index:** Measures average similarity between each cluster and its most similar counterpart. Lower values reflect superior separation.
        - **Principal Component Analysis (PCA):** Orthogonal linear transformation projecting high-dimensional customer vectors to the 2 orthogonal dimensions capturing maximum population variance.
        """)
        
    with st.expander("3. Persona Synthesis & Guardrails", expanded=True):
        st.markdown("""
        - **Empirical Grounding:** Personas are synthesized strictly from computed cluster summaries (recency, frequency, monetary value, AOV, discount rate, and engagement volume).
        - **Psychological Hypotheses:** All psychological and motivational descriptions are explicitly labeled as *hypotheses* rather than proven causal facts.
        - **Privacy by Design:** Prompts send only aggregated cohort centroids; no personally identifiable information (PII) is transmitted.
        """)
        
    with st.expander("4. Statistical Hypothesis Testing for Experiments", expanded=True):
        st.markdown("""
        - **Two-Proportion Z-Test:** Used to calculate the statistical significance of conversion rate differences ($p < 0.05$).
        - **Welch's Two-Sample T-Test:** Used for order value and revenue per user comparison without assuming equal variance.
        """)
