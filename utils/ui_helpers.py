"""
SegmentIQ AI - UI Helpers and Modern SaaS Styling System (Light Theme)
Provides custom CSS injection, KPI card formatters, badge styling, and Plotly light theme styling.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def apply_custom_styles():
    """Injects modern, executive-grade light-theme SaaS CSS styling into the Streamlit app."""
    custom_css = """
    <style>
    /* Global Styling & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* App Base Light Styling */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #1E293B;
    }
    
    /* Metric Cards (Light Premium SaaS) */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 6px 16px -4px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        margin-bottom: 12px;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #2563EB;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1), 0 10px 24px -6px rgba(0, 0, 0, 0.06);
    }
    .metric-title {
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 6px;
    }
    .metric-badge-pos {
        display: inline-block;
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .metric-badge-neg {
        display: inline-block;
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .metric-badge-info {
        display: inline-block;
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Persona Banner Cards */
    .persona-card {
        background: #FFFFFF;
        border-left: 4px solid #2563EB;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    
    /* Strategy Justification Block */
    .justification-card {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 10px;
        color: #1E293B;
    }
    
    /* Subheaders and Accents */
    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    h4, h5, h6 {
        color: #1E293B !important;
        font-weight: 600 !important;
    }
    
    /* DataFrame Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        background-color: #FFFFFF;
    }
    
    /* Tabs & Radio button styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 6px;
        padding: 8px 16px;
        color: #475569;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_metric_card(title: str, value: str, subtitle: str = "", badge: str = "", badge_type: str = "pos"):
    """Renders a styled KPI metric card in HTML."""
    badge_html = ""
    if badge:
        cls = "metric-badge-pos" if badge_type == "pos" else ("metric-badge-neg" if badge_type == "neg" else "metric-badge-info")
        badge_html = f"<span class='{cls}'>{badge}</span>"
        
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{badge_html} {subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def get_plotly_layout_defaults() -> dict:
    """Returns light-themed default layout parameters for Plotly figures."""
    return dict(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#334155"),
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
        yaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
        legend=dict(bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#E2E8F0")
    )
