"""
SegmentIQ AI - Reporting and Multi-Format Export Module
Generates professional PDF management reports via ReportLab and multi-tab Excel workbooks.
"""

import io
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)

def generate_excel_report(
    df_clustered: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    personas_dict: Dict[str, Any],
    strategies_dict: Dict[str, Any],
    metrics: Dict[str, Any]
) -> bytes:
    """
    Creates a multi-tab Excel workbook containing complete intelligence assets.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Customer Assignments
        export_cust_cols = [
            c for c in [
                "customer_id", "segment_code", "cluster_id", "recency_days",
                "purchase_frequency", "monetary_value", "avg_order_value",
                "discount_usage_pct", "total_interactions", "estimated_clv",
                "pca_1", "pca_2"
            ] if c in df_clustered.columns
        ]
        df_clustered[export_cust_cols].to_excel(writer, sheet_name="Customer_Assignments", index=False)
        
        # Sheet 2: Segment Profiles
        cluster_summary_df.to_excel(writer, sheet_name="Segment_Profiles", index=False)
        
        # Sheet 3: Personas
        persona_rows = []
        for code, p in personas_dict.get("personas", {}).items():
            persona_rows.append({
                "Segment Code": code,
                "Persona Title": p.get("persona_title"),
                "Short Description": p.get("short_description"),
                "Dominant Behaviors": " | ".join(p.get("dominant_behaviors", [])),
                "Spending Profile": p.get("spending_profile"),
                "Engagement Pattern": p.get("engagement_pattern"),
                "Hypothesized Needs": " | ".join(p.get("hypothesized_needs", [])),
                "Retention Barriers": " | ".join(p.get("retention_barriers", [])),
                "Confidence": p.get("confidence_assessment")
            })
        pd.DataFrame(persona_rows).to_excel(writer, sheet_name="Customer_Personas", index=False)
        
        # Sheet 4: Marketing Strategies
        strat_rows = []
        for code, s in strategies_dict.items():
            c = s.get("campaign_strategy", {})
            j = s.get("justification", {})
            strat_rows.append({
                "Segment Code": code,
                "Segment Title": s.get("segment_title"),
                "Lifecycle Stage": s.get("lifecycle_stage"),
                "Objective": c.get("objective"),
                "Channel": c.get("channel"),
                "Core Message": c.get("core_message"),
                "Recommended Offer": c.get("incentive"),
                "Timing": c.get("timing"),
                "Expected Impact": j.get("expected_direction_of_impact"),
                "KPI to Monitor": j.get("kpi_to_monitor")
            })
        pd.DataFrame(strat_rows).to_excel(writer, sheet_name="Marketing_Strategies", index=False)
        
        # Sheet 5: Model Evaluation Metrics
        metrics_df = pd.DataFrame([
            {"Metric": k, "Value": str(v)} for k, v in metrics.items() if k != "features_used"
        ])
        metrics_df.to_excel(writer, sheet_name="Clustering_Metrics", index=False)
        
    return output.getvalue()

def generate_pdf_report(
    df_clustered: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    personas_dict: Dict[str, Any],
    strategies_dict: Dict[str, Any],
    metrics: Dict[str, Any],
    overall_kpis: Dict[str, Any]
) -> bytes:
    """
    Builds a professional executive PDF report using ReportLab.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom typography
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=14,
        spaceAfter=8
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1E293B')
    )
    
    table_hdr = ParagraphStyle(
        'TableHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )
    
    story = []
    
    # 1. HEADER BANNER
    story.append(Paragraph("SegmentIQ AI", title_style))
    story.append(Paragraph("LLM-Powered Customer Segmentation & Strategic Marketing Intelligence Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=15))
    
    # 2. EXECUTIVE SUMMARY & KEY KPIS
    story.append(Paragraph("1. Executive Summary & Core KPIs", h1_style))
    story.append(Paragraph(
        f"This intelligence report provides an unsupervised behavioral segmentation analysis of <b>{overall_kpis.get('total_customers', len(df_clustered)):,}</b> "
        f"active customers across <b>${overall_kpis.get('total_revenue', 0.0):,.2f}</b> in completed transaction value. "
        f"Rather than relying on broad demographic proxies (age/gender), the system applied <b>{metrics.get('algorithm', 'K-Means')}</b> clustering "
        f"on normalized multi-dimensional behavioral features, discovering <b>{metrics.get('n_clusters', 5)}</b> distinct customer segments.",
        body_style
    ))
    
    # KPI Table
    kpi_data = [
        [
            Paragraph("<b>Total Customers</b>", table_cell),
            Paragraph(f"{overall_kpis.get('total_customers', len(df_clustered)):,}", table_cell),
            Paragraph("<b>Total Net Revenue</b>", table_cell),
            Paragraph(f"${overall_kpis.get('total_revenue', 0.0):,.2f}", table_cell)
        ],
        [
            Paragraph("<b>Discovered Segments</b>", table_cell),
            Paragraph(f"{metrics.get('n_clusters', 5)}", table_cell),
            Paragraph("<b>Average Order Value</b>", table_cell),
            Paragraph(f"${overall_kpis.get('avg_aov', 0.0):.2f}", table_cell)
        ],
        [
            Paragraph("<b>Avg Purchase Frequency</b>", table_cell),
            Paragraph(f"{overall_kpis.get('avg_frequency', 0.0):.1f} orders", table_cell),
            Paragraph("<b>Repeat Customer Rate</b>", table_cell),
            Paragraph(f"{overall_kpis.get('repeat_rate', 0.0):.1f}%", table_cell)
        ],
        [
            Paragraph("<b>Top 20% Revenue Concentration</b>", table_cell),
            Paragraph(f"{overall_kpis.get('top_20_share', 0.0):.1f}%", table_cell),
            Paragraph("<b>At-Risk Customer Rate</b>", table_cell),
            Paragraph(f"{overall_kpis.get('at_risk_rate', 0.0):.1f}%", table_cell)
        ]
    ]
    
    t_kpi = Table(kpi_data, colWidths=[130, 130, 140, 140])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 12))
    
    # 3. SEGMENTATION EVALUATION & MODEL METRICS
    story.append(Paragraph("2. Machine Learning Segmentation Quality", h1_style))
    story.append(Paragraph(
        f"<b>Algorithm:</b> {metrics.get('algorithm', 'K-Means')} | "
        f"<b>Silhouette Score:</b> {metrics.get('silhouette_score', 'N/A')} | "
        f"<b>Davies-Bouldin Index:</b> {metrics.get('davies_bouldin_index', 'N/A')} | "
        f"<b>Calinski-Harabasz Index:</b> {metrics.get('calinski_harabasz_index', 'N/A')} | "
        f"<b>PCA Variance Explained:</b> {metrics.get('pca_total_variance', 'N/A')}% (2 components)",
        body_style
    ))
    story.append(Spacer(1, 8))
    
    # Segment Summary Table
    story.append(Paragraph("<b>Discovered Segment Profiles:</b>", body_style))
    seg_tbl_data = [[
        Paragraph("<b>Segment</b>", table_hdr),
        Paragraph("<b>Customers</b>", table_hdr),
        Paragraph("<b>Share %</b>", table_hdr),
        Paragraph("<b>Rev Share %</b>", table_hdr),
        Paragraph("<b>Avg Recency</b>", table_hdr),
        Paragraph("<b>Avg Freq</b>", table_hdr),
        Paragraph("<b>Avg AOV</b>", table_hdr)
    ]]
    
    for _, row in cluster_summary_df.iterrows():
        seg_tbl_data.append([
            Paragraph(str(row["segment_code"]), table_cell),
            Paragraph(f"{int(row.get('customer_count', 0)):,}", table_cell),
            Paragraph(f"{row.get('pct_of_customers', 0.0):.1f}%", table_cell),
            Paragraph(f"{row.get('revenue_share_pct', 0.0):.1f}%", table_cell),
            Paragraph(f"{row.get('recency_days', 0.0):.0f}d", table_cell),
            Paragraph(f"{row.get('purchase_frequency', 0.0):.1f}", table_cell),
            Paragraph(f"${row.get('avg_order_value', 0.0):.2f}", table_cell)
        ])
        
    t_seg = Table(seg_tbl_data, colWidths=[90, 70, 60, 75, 75, 75, 95])
    t_seg.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_seg)
    story.append(Spacer(1, 14))
    
    # 4. CUSTOMER PERSONAS & STRATEGIES
    story.append(Paragraph("3. Customer Personas & Differentiated Strategies", h1_style))
    personas = personas_dict.get("personas", {})
    
    for code, p in personas.items():
        title = p.get("persona_title", code)
        strat = strategies_dict.get(code, {})
        camp = strat.get("campaign_strategy", {})
        just = strat.get("justification", {})
        
        story.append(Paragraph(f"<b>{code}: {title}</b>", h2_style))
        story.append(Paragraph(f"<b>Overview:</b> {p.get('short_description', '')}", body_style))
        story.append(Paragraph(f"<b>Spending & Engagement:</b> {p.get('spending_profile', '')} {p.get('engagement_pattern', '')}", body_style))
        story.append(Paragraph(f"<b>Lifecycle Objective:</b> {strat.get('lifecycle_stage', '')} — {camp.get('objective', '')}", body_style))
        story.append(Paragraph(f"<b>Recommended Action:</b> Channel: {camp.get('channel', '')} | Offer: {camp.get('incentive', '')}", body_style))
        story.append(Paragraph(f"<b>KPI to Monitor:</b> {just.get('kpi_to_monitor', 'Conversion & Retention')}", body_style))
        story.append(Spacer(1, 6))
        
    # 5. GOVERNANCE & LIMITATIONS
    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Data Limitations & Implementation Next Steps", h1_style))
    story.append(Paragraph(
        "1. <b>Hypotheses vs Causality:</b> Behavioral clusters reveal empirical correlation patterns, not proven psychological causes. "
        "All persona motivations are explicitly labeled as hypotheses to be verified.<br/>"
        "2. <b>Experimental Validation:</b> High-impact strategies (such as loyalty perks or margin-protection discounts) "
        "should be tested via controlled A/B experiments before full roll-out.<br/>"
        "3. <b>Model Retraining:</b> Clustering models should be refreshed monthly or quarterly to adapt to changing customer behaviors.",
        body_style
    ))
    
    doc.build(story)
    return buffer.getvalue()
