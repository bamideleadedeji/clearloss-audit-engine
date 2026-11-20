import io
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ingestion import SEC_COMPANIES, fetch_live_sec_filings, standardize_uploaded_ledger
from forensic_engine import analyze_benfords_law, run_anomaly_detection
from split_invoice_detector import detect_split_invoices, generate_split_summary

st.set_page_config(
    page_title="ClearLoss Enterprise | Forensic Revenue Assurance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def build_excel_workbook(full_df, flagged_df, split_summary, benford_res, source_label, limit_amt):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_rows = [
            {"Metric": "Audit Timestamp (UTC)", "Value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")},
            {"Metric": "Data Source", "Value": source_label},
            {"Metric": "Approval Threshold Tested ($)", "Value": f"${limit_amt:,.2f}"},
            {"Metric": "Total Audited Transactions", "Value": int(len(full_df))},
            {"Metric": "Total Audited Volume ($)", "Value": f"${full_df['amount'].sum():,.2f}"},
            {"Metric": "High-Risk Flagged Items (Score >= 60)", "Value": int(len(flagged_df))},
            {"Metric": "Flagged Exposure Volume ($)", "Value": f"${flagged_df['amount'].sum():,.2f}"},
            {"Metric": "Split-Invoice Clusters Detected", "Value": int(len(split_summary))},
            {"Metric": "Benford MAD Score", "Value": f"{benford_res.get('mad', 0.0):.5f} ({benford_res.get('mad_status', 'N/A')})"},
            {"Metric": "Benford Chi-Square (p-value)", "Value": f"{benford_res.get('chi_stat', 0.0):.2f} (p={benford_res.get('p_value', 1.0):.4f})"},
        ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="01_Executive_Summary", index=False)

        if not split_summary.empty:
            split_summary.to_excel(writer, sheet_name="02_Split_Invoice_Clusters", index=False)

        want_cols = [
            "transaction_id", "vendor_name", "agency", "amount", "date",
            "combined_risk_score", "audit_finding", "cluster_id", "vendor_zscore"
        ]
        f_cols = [c for c in want_cols if c in flagged_df.columns]
        flagged_df[f_cols].to_excel(writer, sheet_name="03_Flagged_Transactions", index=False)

        all_cols = [c for c in want_cols if c in full_df.columns]
        full_df[all_cols].to_excel(writer, sheet_name="04_Full_Scored_Ledger", index=False)

    return output.getvalue()
def build_forensic_memo(full_df, flagged_df, split_summary, benford_res, source_label, limit_amt):
    total_spend = float(full_df["amount"].sum())
    flagged_spend = float(flagged_df["amount"].sum())
    leakage_pct = (flagged_spend / total_spend * 100.0) if total_spend > 0 else 0.0
    lines = [
        "=====================================================================",
        "CLEARLOSS ENTERPRISE — FORENSIC AUDIT & REVENUE ASSURANCE MEMORANDUM",
        "=====================================================================",
        f"Generated (UTC)         : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Ledger Data Source      : {source_label}",
        f"Approval Limit Tested   : ${limit_amt:,.2f}",
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
        "   - Request supporting purchase orders, receiving reports, and dual-signoff",
        "     approvals for all split-invoice clusters bypassing the approval limit.",
        "   - Hold disbursement or initiate recovery on exact duplicate payments (<7d).",
        "   - Perform substantive vouching on round-dollar invoices and >3-sigma outliers.",
        "=====================================================================",
    ]
    return "\n".join(lines)


@st.cache_data(ttl=1800)
def cached_sec_filings(company_label: str, limit: int):
    return fetch_live_sec_filings(company_label=company_label, limit=limit)


st.title("🛡️ ClearLoss™ Enterprise Revenue Assurance & Forensic Audit Engine")
st.caption("100% Real-Data Forensic Suite • Benford's Law & Nigrini MAD • Isolation Forest ML • Duplicate & Split-Invoice Controls")

st.sidebar.header("⚙️ Audit Engine Controls")
data_mode = st.sidebar.radio(
    "Select Verified Data Source",
    ["🌐 Live US SEC EDGAR Audited Filings", "📁 Upload Client CSV / Excel Ledger"],
)

uploaded_file = None
sec_company = "Apple Inc. (AAPL)"
record_limit = 500

if data_mode == "🌐 Live US SEC EDGAR Audited Filings":
    sec_company = st.sidebar.selectbox("Public Registrant (SEC EDGAR)", list(SEC_COMPANIES.keys()))
    record_limit = st.sidebar.slider("SEC Filing Line Items Limit", 100, 1500, 500, 100)
if data_mode == "📁 Upload Client CSV / Excel Ledger":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Accounts Payable / Procurement Ledger",
        type=["csv", "xlsx", "xls"],
    )

threshold_limit = st.sidebar.number_input("Single-Transaction Approval Limit ($)", 500.0, 1000000.0, 10000.0, 500.0)
window_hours = st.sidebar.slider("Split-Invoice Rolling Window (Hours)", 12, 168, 48, 12)

if st.sidebar.button("🔄 Refresh Audit Pipeline"):
    st.cache_data.clear()
    st.rerun()
raw_df = None
source_label = ""

if data_mode == "🌐 Live US SEC EDGAR Audited Filings":
    try:
        with st.spinner(f"Streaming audited US-GAAP filings for {sec_company} from SEC EDGAR..."):
            raw_df, source_label = cached_sec_filings(sec_company, int(record_limit))
    except Exception as exc:
        st.error(f"SEC EDGAR API connection error: {exc}")
        st.stop()

if data_mode == "📁 Upload Client CSV / Excel Ledger":
    if uploaded_file is None:
        st.info("👈 Upload a client **CSV or Excel** Accounts Payable ledger in the sidebar to run your forensic audit (zero simulated data is ever used).")
        st.stop()
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded_file)
        if not uploaded_file.name.lower().endswith(".csv"):
            df_in = pd.read_excel(uploaded_file)
        raw_df = standardize_uploaded_ledger(df_in)
        source_label = f"CLIENT LEDGER: {uploaded_file.name} ({len(raw_df):,} verified records)"
    except Exception as exc:
        st.error(f"Could not parse uploaded ledger: {exc}")
        st.stop()

st.success(f"✅ **Verified Audit Source:** {source_label}")

ml_df = run_anomaly_detection(raw_df)
df = detect_split_invoices(
    ml_df,
    approval_threshold=float(threshold_limit),
    window_hours=int(window_hours),
    min_transactions=2,
)
df["combined_risk_score"] = np.maximum(df["risk_score"], df["split_risk_score"])

reasons = []
for _, r in df.iterrows():
    tags = []
    if r.get("split_invoice_flag", False):
        tags.append(f"Split-Invoice Evasion ({r.get('cluster_id', 'CLUSTER')})")
    if r.get("flagged_reason", "Normal") != "Normal":
        tags.append(str(r["flagged_reason"]))
    reasons.append(" + ".join(tags) if tags else "Normal")
df["audit_finding"] = reasons

split_summary = generate_split_summary(df, approval_threshold=float(threshold_limit))
benford_res = analyze_benfords_law(df)

total_spend = float(df["amount"].sum())
high_risk_df = df[df["combined_risk_score"] >= 60.0].sort_values(by="combined_risk_score", ascending=False)
total_leakage = float(high_risk_df["amount"].sum())
leakage_pct = (total_leakage / total_spend * 100.0) if total_spend > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Audited Volume", f"${total_spend:,.2f}")
c2.metric("Flagged Exposure Volume", f"${total_leakage:,.2f}", delta=f"{leakage_pct:.1f}% of Spend", delta_color="inverse")
c3.metric("Audited Transactions", f"{len(df):,}")
c4.metric("High-Risk Flagged Items", f"{len(high_risk_df):,}")

st.markdown("---")
t1, t2, t3, t4 = st.tabs([
    "📊 Benford's Law & Nigrini MAD",
    "🤖 ML & Duplicate Controls",
    "🚨 Split-Invoice Evasion",
    "📑 Executive Ledger & Exports",
])

with t1:
    st.subheader("First-Digit Logarithmic Distribution & Nigrini MAD Conformity")
    if benford_res:
        b1, b2, b3 = st.columns(3)
        b1.metric("Nigrini MAD Score", f"{benford_res['mad']:.5f}")
        b2.metric("Conformity Rating", benford_res["mad_status"])
        b3.metric("Chi-Square (p-value)", f"{benford_res['chi_stat']:.2f} (p={benford_res['p_value']:.4f})")

        fig_ben = go.Figure()
        fig_ben.add_trace(go.Bar(x=benford_res["digit"], y=benford_res["observed"], name="Observed"))
        fig_ben.add_trace(go.Scatter(x=benford_res["digit"], y=benford_res["expected"], name="Benford Expected", line=dict(color="red", dash="dash")))
        fig_ben.update_layout(xaxis=dict(title="Leading Digit (1–9)", tickmode="linear"), yaxis_title="Proportion", height=380)
        st.plotly_chart(fig_ben, use_container_width=True)

        if benford_res.get("two_digit_spikes"):
            st.warning("⚠️ First-Two-Digit (10–99) Threshold Clustering Detected (Z >= 2.58):")
            st.dataframe(pd.DataFrame(benford_res["two_digit_spikes"]), use_container_width=True)

with t2:
    st.subheader("Isolation Forest Outliers, Duplicate Payments & Round-Dollar Controls")
    fig_sc = px.scatter(
        df, x="amount", y="combined_risk_score", color="audit_finding",
        hover_data=["vendor_name", "agency", "transaction_id"], log_x=True,
        title="Transaction Amount vs. Composite Forensic Risk Score",
    )
    fig_sc.update_layout(height=440)
    st.plotly_chart(fig_sc, use_container_width=True)

with t3:
    st.subheader(f"Split-Invoice Threshold Evasion Clusters (< ${threshold_limit:,.2f} within {window_hours}h)")
    if not split_summary.empty:
        s1, s2 = st.columns(2)
        s1.metric("Flagged Split Clusters", len(split_summary))
        s2.metric("Total Bypassed Audit Spend", f"${split_summary['total_split_amount'].sum():,.2f}")
        st.dataframe(split_summary, use_container_width=True)
    if split_summary.empty:
        st.info(f"No split-invoice threshold evasions detected under ${threshold_limit:,.2f} within {window_hours} hours.")

with t4:
    st.subheader("High-Risk Audit Findings & Client Deliverables")
    min_score = st.slider("Filter Minimum Risk Score", 0, 100, 60)
    filt = df[df["combined_risk_score"] >= min_score].sort_values(by="combined_risk_score", ascending=False)
    show_cols = [c for c in ["transaction_id", "vendor_name", "agency", "amount", "combined_risk_score", "audit_finding", "cluster_id", "date"] if c in filt.columns]
    st.dataframe(filt[show_cols], use_container_width=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        xb = build_excel_workbook(df, high_risk_df, split_summary, benford_res, source_label, float(threshold_limit))
        st.download_button("📊 Download Multi-Sheet Excel Audit (.xlsx)", xb, "ClearLoss_Enterprise_Audit_Report.xlsx")
    with d2:
        mt = build_forensic_memo(df, high_risk_df, split_summary, benford_res, source_label, float(threshold_limit))
        st.download_button("📝 Download CFO Forensic Memo (.txt)", mt.encode("utf-8"), "ClearLoss_CFO_Forensic_Memo.txt")
    with d3:
        cd = filt[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Flagged Ledger (.csv)", cd, "ClearLoss_Flagged_Transactions.csv")
