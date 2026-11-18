import io
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ingestion import fetch_live_procurement_data, standardize_uploaded_ledger
from forensic_engine import analyze_benfords_law, run_anomaly_detection
from split_invoice_detector import detect_split_invoices, generate_split_summary

st.set_page_config(
    page_title="ClearLoss Enterprise | Forensic Revenue Assurance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ ClearLoss™ Enterprise Revenue Assurance & Forensic Audit Engine")
st.markdown(
    "**Institutional Audit Intelligence Suite** • Benford's Law & Nigrini MAD Conformity • "
    "Isolation Forest ML • Duplicate & Round-Dollar Controls • Split-Invoice Threshold Evasion"
)


def build_excel_audit_workbook(
    full_df: pd.DataFrame,
    flagged_df: pd.DataFrame,
    split_summary: pd.DataFrame,
    benford_res: dict,
    source_label: str,
    threshold_limit: float,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        exec_summary = pd.DataFrame([
            {"Metric": "Audit Timestamp (UTC)", "Value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")},
            {"Metric": "Data Source", "Value": source_label},
            {"Metric": "Approval Threshold Tested ($)", "Value": f"${threshold_limit:,.2f}"},
            {"Metric": "Total Audited Transactions", "Value": int(len(full_df))},
            {"Metric": "Total Ingested Spend ($)", "Value": f"${full_df['amount'].sum():,.2f}"},
            {"Metric": "High-Risk Flagged Items (Score >= 60)", "Value": int(len(flagged_df))},
            {"Metric": "Total Flagged Exposure ($)", "Value": f"${flagged_df['amount'].sum():,.2f}"},
            {"Metric": "Split-Invoice Clusters Detected", "Value": int(len(split_summary))},
            {"Metric": "Benford MAD Score", "Value": f"{benford_res.get('mad', 0.0):.5f} ({benford_res.get('mad_status', 'N/A')})"},
            {"Metric": "Benford Chi-Square (p-value)", "Value": f"{benford_res.get('chi_stat', 0.0):.2f} (p={benford_res.get('p_value', 1.0):.4f})"},
        ])
        exec_summary.to_excel(writer, sheet_name="01_Executive_Summary", index=False)

        if not split_summary.empty:
            split_summary.to_excel(writer, sheet_name="02_Split_Invoice_Clusters", index=False)

        export_cols = [
            c for c in [
                "transaction_id", "vendor_name", "agency", "amount", "date",
                "combined_risk_score", "audit_finding", "cluster_id", "vendor_zscore"
            ] if c in flagged_df.columns
        ]
        flagged_df[export_cols].to_excel(writer, sheet_name="03_Flagged_Transactions", index=False)

        full_cols = [
            c for c in [
                "transaction_id", "vendor_name", "agency", "amount", "date",
                "combined_risk_score", "audit_finding", "cluster_id"
            ] if c in full_df.columns
        ]
        full_df[full_cols].to_excel(writer, sheet_name="04_Full_Scored_Ledger", index=False)

    return output.getvalue()


def build_forensic_memo(
    full_df: pd.DataFrame,
    flagged_df: pd.DataFrame,
    split_summary: pd.DataFrame,
    benford_res: dict,
    source_label: str,
    threshold_limit: float,
) -> str:
    total_spend = float(full_df["amount"].sum())
    flagged_spend = float(flagged_df["amount"].sum())
    leakage_pct = (flagged_spend / total_spend * 100.0) if total_spend > 0 else 0.0
    lines = [
        "=====================================================================",
        "CLEARLOSS ENTERPRISE — FORENSIC AUDIT & REVENUE ASSURANCE MEMORANDUM",
        "=====================================================================",
        f"Generated (UTC)         : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Ledger Data Source      : {source_label}",
        f"Approval Limit Tested   : ${threshold_limit:,.2f}",
        "---------------------------------------------------------------------",
        "1. EXECUTIVE FINANCIAL EXPOSURE SUMMARY",
        f"   - Total Audited Transactions : {len(full_df):,}",
        f"   - Total Ledger Spend         : ${total_spend:,.2f}",
        f"   - Flagged High-Risk Items    : {len(flagged_df):,}",
        f"   - Flagged Exposure Volume    : ${flagged_spend:,.2f} ({leakage_pct:.2f}% of spend)",
        f"   - Split-Invoice Clusters     : {len(split_summary):,}",
        "",
        "2. BENFORD'S LAW & NIGRINI CONFORMITY ASSESSMENT",
        f"   - Mean Absolute Deviation    : {benford_res.get('mad', 0.0):.5f}",
        f"   - Conformity Classification  : {benford_res.get('mad_status', 'N/A')}",
        f"   - Chi-Square Statistic       : {benford_res.get('chi_stat', 0.0):.2f} (p = {benford_res.get('p_value', 1.0):.4f})",
        "",
        "3. RECOMMENDED AUDIT COMMITTEE ACTIONS",
        "   - Request original supporting POs, receiving reports, and dual-signoff approvals",
        "     for all split-invoice clusters bypassing the single-transaction limit.",
        "   - Hold disbursement or initiate vendor recovery on exact duplicate payments (<7 days).",
        "   - Perform substantive vouching on round-dollar invoices and >3-sigma vendor outliers.",
        "=====================================================================",
    ]
    return "\n".join(lines)


st.sidebar.header("⚙️ Audit Engine Controls")
data_mode = st.sidebar.radio(
    "Select Ledger Source",
    ["📁 Upload Client CSV / Excel Ledger", "🌐 Live USAspending.gov Federal API"],
)

uploaded_file = None
record_limit = 300
if data_mode == "📁 Upload Client CSV / Excel Ledger":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Accounts Payable / Procurement Ledger",
        type=["csv", "xlsx", "xls"],
        help="Auto-maps Amount, Vendor, Date, Invoice ID, and Department columns.",
    )
else:
    record_limit = st.sidebar.slider("USAspending API Query Limit", min_value=100, max_value=1000, value=300, step=100)

threshold_limit = st.sidebar.number_input("Single-Transaction Approval Limit ($)", min_value=500.0, value=10000.0, step=500.0)
window_hours = st.sidebar.slider("Split-Invoice Rolling Window (Hours)", min_value=12, max_value=168, value=48, step=12)

if st.sidebar.button("🔄 Clear Cache & Re-Run Audit"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(ttl=1800)
def get_live_api_batch(limit: int):
    return fetch_live_procurement_data(limit=limit)


raw_df = None
source_label = ""

if data_mode == "📁 Upload Client CSV / Excel Ledger":
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded_file)
            else:
                df_in = pd.read_excel(uploaded_file)
            raw_df = standardize_uploaded_ledger(df_in)
            source_label = f"CLIENT LEDGER: {uploaded_file.name} ({len(raw_df):,} records)"
        except Exception as exc:
            st.error(f"Could not parse uploaded ledger: {exc}")
            st.stop()
    else:
        st.info("👈 Upload a client **CSV or Excel** file in the sidebar, or switch to **Live USAspending.gov Federal API** to audit real federal contract awards immediately.")
        raw_df, _ = get_live_api_batch(200)
        source_label = f"PREVIEW SAMPLE LEDGER ({len(raw_df):,} records — upload a client file anytime)"
else:
    with st.spinner("Streaming verified contract awards from USAspending.gov..."):
        raw_df, source_label = get_live_api_batch(record_limit)

if "OFFLINE DEMO" in source_label:
    st.warning(f"⚠️ **Data Source Notice:** {source_label}")
else:
    st.success(f"✅ **Active Audit Source:** {source_label}")

ml_df = run_anomaly_detection(raw_df)
df = detect_split_invoices(
    ml_df,
    approval_threshold=float(threshold_limit),
    window_hours=int(window_hours),
    min_transactions=2,
)

df["combined_risk_score"] = np.maximum(df["risk_score"], df["split_risk_score"])

combined_reasons = []
for _, r in df.iterrows():
    parts = []
    if r.get("split_invoice_flag", False):
        parts.append(f"Split-Invoice Evasion ({r.get('cluster_id', 'CLUSTER')})")
    if r.get("flagged_reason", "Normal") != "Normal":
        parts.append(str(r["flagged_reason"]))
    combined_reasons.append(" + ".join(parts) if parts else "Normal")
df["audit_finding"] = combined_reasons

split_summary = generate_split_summary(df, approval_threshold=float(threshold_limit))
benford_res = analyze_benfords_law(df)

total_spend = float(df["amount"].sum())
high_risk_df = df[df["combined_risk_score"] >= 60.0].sort_values(by="combined_risk_score", ascending=False)
total_leakage = float(high_risk_df["amount"].sum())
leakage_pct = (total_leakage / total_spend * 100.0) if total_spend > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Audited Volume", f"${total_spend:,.2f}")
col2.metric(
    "Flagged Exposure Volume",
    f"${total_leakage:,.2f}",
    delta=f"{leakage_pct:.1f}% of Spend",
    delta_color="inverse",
)
col3.metric("Audited Transactions", f"{len(df):,}")
col4.metric("High-Risk Flagged Items", f"{len(high_risk_df):,}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Benford's Law & Nigrini MAD",
    "🤖 ML & Duplicate Controls",
    "🚨 Split-Invoice Evasion",
    "📑 Executive Ledger & Exports",
])

with tab1:
    st.subheader("First-Digit Logarithmic Distribution & Nigrini MAD Conformity")
    if benford_res:
        b1, b2, b3 = st.columns(3)
        b1.metric("Nigrini MAD Score", f"{benford_res['mad']:.5f}")
        b2.metric("Conformity Rating", benford_res["mad_status"])
        b3.metric("Chi-Square (p-value)", f"{benford_res['chi_stat']:.2f} (p={benford_res['p_value']:.4f})")

        fig_ben = go.Figure()
        fig_ben.add_trace(go.Bar(
            x=benford_res["digit"],
            y=benford_res["observed"],
            name="Observed Frequency",
            marker_color="#1f77b4",
        ))
        fig_ben.add_trace(go.Scatter(
            x=benford_res["digit"],
            y=benford_res["expected"],
            name="Benford Standard",
            line=dict(color="#ff4b4b", width=3, dash="dash"),
        ))
        fig_ben.update_layout(
            xaxis=dict(title="Leading Digit (1–9)", tickmode="linear"),
            yaxis_title="Proportion",
            height=400,
        )
        st.plotly_chart(fig_ben, use_container_width=True)

        if benford_res.get("two_digit_spikes"):
            st.warning("⚠️ **First-Two-Digit (10–99) Threshold Clustering Detected (Z >= 2.58):**")
            st.dataframe(pd.DataFrame(benford_res["two_digit_spikes"]), use_container_width=True)
        else:
            st.info("No abnormal two-digit threshold clustering (e.g., 98/99 spikes) detected.")

with tab2:
    st.subheader("Isolation Forest Outliers, Duplicate Payments & Round-Dollar Controls")
    fig_scatter = px.scatter(
        df,
        x="amount",
        y="combined_risk_score",
        color="audit_finding",
        size="amount",
        hover_data=["vendor_name", "agency", "transaction_id"],
        log_x=True,
        title="Transaction Amount vs. Composite Forensic Risk Score",
    )
    fig_scatter.update_layout(height=460, xaxis_title="Transaction Amount ($ - Log Scale)", yaxis_title="Risk Score (0–100)")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader(f"Split-Invoice Threshold Evasion Clusters (< ${threshold_limit:,.2f} within {window_hours}h)")
    if not split_summary.empty:
        s1, s2 = st.columns(2)
        s1.metric("Flagged Vendor Split Clusters", len(split_summary))
        s2.metric("Total Bypassed Audit Spend", f"${split_summary['total_split_amount'].sum():,.2f}")
        st.dataframe(
            split_summary.style.format({
                "total_split_amount": "${:,.2f}",
                "bypassed_threshold": "${:,.2f}",
                "potential_leakage": "${:,.2f}",
            }),
            use_container_width=True,
        )
    else:
        st.info(f"No split-invoice threshold evasions detected under ${threshold_limit:,.2f} within {window_hours} hours.")

with tab4:
    st.subheader("High-Risk Audit Findings & Client Deliverables")
    min_score = st.slider("Filter Minimum Composite Risk Score", min_value=0, max_value=100, value=60)
    filtered_ledger = df[df["combined_risk_score"] >= min_score].sort_values(by="combined_risk_score", ascending=False)

    display_cols = [
        c for c in [
            "transaction_id", "vendor_name", "agency", "amount",
            "combined_risk_score", "audit_finding", "cluster_id", "date"
        ] if c in filtered_ledger.columns
    ]
    st.dataframe(
        filtered_ledger[display_cols].style.format({"amount": "${:,.2f}", "combined_risk_score": "{:.1f}"}),
        use_container_width=True,
    )

    d1, d2, d3 = st.columns(3)
    with d1:
        excel_bytes = build_excel_audit_workbook(
            df, high_risk_df, split_summary, benford_res, source_label, float(threshold_limit)
        )
        st.download_button(
            label="📊 Download Multi-Sheet Excel Audit (.xlsx)",
            data=excel_bytes,
            file_name="ClearLoss_Enterprise_Audit_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with d2:
        memo_text = build_forensic_memo(
            df, high_risk_df, split_summary, benford_res, source_label, float(threshold_limit)
        )
        st.download_button(
            label="📝 Download CFO Forensic Memo (.txt)",
            data=memo_text.encode("utf-8"),
            file_name="ClearLoss_CFO_Forensic_Memo.txt",
            mime="text/plain",
        )
    with d3:
        csv_data = filtered_ledger[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Flagged Ledger (.csv)",
            data=csv_data,
            file_name="ClearLoss_Flagged_Transactions.csv",
            mime="text/csv",
        )
