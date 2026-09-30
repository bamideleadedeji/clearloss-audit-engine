import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest

# Imports from local modules
from ingestion import fetch_live_procurement_data
from forensic_engine import analyze_benfords_law, run_anomaly_detection
from split_invoice_detector import detect_split_invoices, generate_split_summary

# -----------------------------------------------------------------------------
# 1. Page & UI Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ClearLoss | Forensic Revenue Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ ClearLoss: Executive Revenue Assurance Engine")
st.markdown(
    "**Live Enterprise Intelligence Suite** | Real-Time USAspending API Ingestion • "
    "Benford's Law Audit • Isolation Forest ML • Split-Invoice Threshold Analytics"
)

# -----------------------------------------------------------------------------
# 2. Data Loading & Caching Pipeline
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_and_process_data(record_limit: int = 1000):
    # Step 1: Ingest Live Data
    raw_df = fetch_live_procurement_data(limit=record_limit)
    
    # Step 2: Run Machine Learning & Statistical Anomaly Detection
    ml_df = run_anomaly_detection(raw_df)
    
    # Step 3: Run Sliding-Window Split-Invoice Detection ($10k threshold, 48h window)
    full_df = detect_split_invoices(
        ml_df, 
        approval_threshold=10000.0, 
        window_hours=48, 
        min_transactions=2
    )
    
    # Step 4: Generate Consolidated Risk Metrics
    full_df["combined_risk_score"] = np.maximum(
        full_df["risk_score"], 
        full_df["split_risk_score"]
    )
    
    return full_df

# Sidebar Controls
st.sidebar.header("⚙️ Audit Engine Controls")
record_limit = st.sidebar.slider("API Query Ingestion Limit", min_value=200, max_value=2000, value=1000, step=100)
threshold_limit = st.sidebar.number_input("Approval Limit Threshold ($)", value=10000.0, step=1000.0)

if st.sidebar.button("🔄 Refresh API Pipeline"):
    st.cache_data.clear()
    st.rerun()

# Load Data
with st.spinner("Streaming live USAspending transaction records..."):
    df = load_and_process_data(record_limit)

split_summary = generate_split_summary(df, approval_threshold=threshold_limit)

# -----------------------------------------------------------------------------
# 3. C-Suite KPI Metric Bar
# -----------------------------------------------------------------------------
total_spend = df["amount"].sum()
high_risk_df = df[df["combined_risk_score"] > 60]
total_leakage = high_risk_df["amount"].sum()
leakage_pct = (total_leakage / total_spend) * 100 if total_spend > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Ingested Volume", f"${total_spend:,.2f}")
col2.metric(
    "Flagged Leakage Exposure", 
    f"${total_leakage:,.2f}", 
    delta=f"{leakage_pct:.1f}% Total Spend", 
    delta_color="inverse"
)
col3.metric("Audited Transactions", f"{len(df):,}")
col4.metric("High-Risk Flagged Items", f"{len(high_risk_df):,}")

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. Interactive Analytics Workspace Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Benford's Law Audit", 
    "🤖 ML Anomaly Matrix", 
    "🚨 Split-Invoice Evasion", 
    "📑 Detailed Audit Ledger"
])

# -----------------------------------------------------------------------------
# TAB 1: Benford's Law Analysis
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("First-Digit Logarithmic Distribution Test (Benford's Law)")
    st.markdown("Evaluates leading digits against natural logarithmic expectations to detect artificial pricing manipulation.")
    
    benford_res = analyze_benfords_law(df)
    
    if benford_res:
        fig_ben = go.Figure()
        fig_ben.add_trace(go.Bar(
            x=benford_res["digit"], 
            y=benford_res["observed"], 
            name="Observed Distribution",
            marker_color="#1f77b4"
        ))
        fig_ben.add_trace(go.Scatter(
            x=benford_res["digit"], 
            y=benford_res["expected"], 
            name="Benford Standard", 
            line=dict(color='red', width=3, dash='dash')
        ))
        fig_ben.update_layout(
            xaxis=dict(title="First Digit (1-9)", tickmode='linear'),
            yaxis_title="Frequency Proportion",
            height=420,
            legend=dict(x=0.8, y=1)
        )
        st.plotly_chart(fig_ben, use_container_width=True)
        
        stat_col1, stat_col2 = st.columns(2)
        stat_col1.caption(rf"**Chi-Square Statistic ($\chi^2$):** {benford_res['chi_stat']:.2f}")
        
        p_val = benford_res['p_value']
        status_msg = "⚠️ High Anomaly Variance Detected (p < 0.05)" if p_val < 0.05 else "✅ Natural Distribution Pattern (p >= 0.05)"
        stat_col2.caption(f"**Statistical p-value:** {p_val:.4f} — *{status_msg}*")

# -----------------------------------------------------------------------------
# TAB 2: Isolation Forest & Multivariate ML Matrix
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Multivariate Isolation Forest Anomaly Matrix")
    st.markdown("Unsupervised machine learning scoring based on transaction volume, velocity, and vendor variance.")
    
    fig_scatter = px.scatter(
        df, 
        x="amount", 
        y="combined_risk_score", 
        color="flagged_reason",
        size="amount",
        hover_data=["vendor_name", "agency", "transaction_id"],
        log_x=True,
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Transaction Amount vs. Combined Risk Score"
    )
    fig_scatter.update_layout(height=480, xaxis_title="Award Amount ($ - Log Scale)", yaxis_title="Calculated Risk Score (0 - 100)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Split-Invoice Evasion Detection
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Split-Invoice Threshold Evasion Clusters")
    st.markdown(f"Identifies vendors issuing multiple sub-threshold payments (e.g., under ${threshold_limit:,.2f}) within 48-hour rolling windows.")
    
    if not split_summary.empty:
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Flagged Vendor Clusters", len(split_summary))
        col_s2.metric("Total Bypassed Audit Value", f"${split_summary['total_split_amount'].sum():,.2f}")
        
        st.dataframe(
            split_summary.style.format({
                "total_split_amount": "${:,.2f}",
                "bypassed_threshold": "${:,.2f}",
                "potential_leakage": "${:,.2f}"
            }), 
            use_container_width=True
        )
    else:
        st.info("No split-invoice threshold evasions detected in the current query batch.")

# -----------------------------------------------------------------------------
# TAB 4: Comprehensive Line-Item Audit Ledger
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("High-Risk Line-Item Audit Ledger")
    
    min_score = st.slider("Filter Minimum Risk Score", min_value=0, max_value=100, value=50)
    filtered_ledger = df[df["combined_risk_score"] >= min_score].sort_values(by="combined_risk_score", ascending=False)
    
    st.dataframe(
        filtered_ledger[[
            "transaction_id", "vendor_name", "agency", "amount", 
            "combined_risk_score", "flagged_reason", "date"
        ]].style.format({"amount": "${:,.2f}", "combined_risk_score": "{:.1f}"}),
        use_container_width=True
    )
    
    # Export Capabilities
    csv_data = filtered_ledger.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Audit Findings (CSV)",
        data=csv_data,
        file_name="ClearLoss_Audit_Findings.csv",
        mime="text/csv"
    )
