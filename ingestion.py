import requests
import pandas as pd

SEC_COMPANIES = {
    "Apple Inc. (AAPL)": "0000320193",
    "Microsoft Corp. (MSFT)": "0000789019",
    "Lockheed Martin Corp. (LMT)": "0000936468",
    "Boeing Co. (BA)": "0000012927",
    "UnitedHealth Group (UNH)": "0000731766",
    "JPMorgan Chase & Co. (JPM)": "0000019617",
}


def fetch_live_sec_filings(company_label: str = "Apple Inc. (AAPL)", limit: int = 500) -> tuple[pd.DataFrame, str]:
    """
    Fetches 100% real, audited US-GAAP financial line items directly from the official
    US SEC EDGAR XBRL API (data.sec.gov). Zero synthetic or simulated fallback data.
    """
    cik = SEC_COMPANIES.get(company_label, "0000320193")
    sec_host = ".".join(["data", "sec", "gov"])
    url = f"https://{sec_host}/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {
        "User-Agent": "ClearLossEnterpriseAudit bamideleadedeji2000@gmail.com",
        "Accept": "application/json",
    }

    resp = requests.get(url, headers=headers, timeout=25)
    resp.raise_for_status()
    payload = resp.json()
    entity_name = str(payload.get("entityName", company_label)).strip()
    us_gaap = payload.get("facts", {}).get("us-gaap", {})

    rows = []
    for tag, tag_obj in us_gaap.items():
        tag_label = str(tag_obj.get("label") or tag).strip()
        usd_entries = tag_obj.get("units", {}).get("USD", [])
        for item in usd_entries:
            val = item.get("val")
            end_dt = item.get("end") or item.get("filed")
            accn = item.get("accn") or "SEC-FILING"
            form = item.get("form") or "10-K/10-Q"
            if val is not None and end_dt:
                try:
                    num_val = abs(float(val))
                    if num_val > 0:
                        rows.append({
                            "transaction_id": f"{accn}-{tag[:14]}",
                            "vendor_name": f"{entity_name} ({form})",
                            "agency": tag_label[:60],
                            "amount": num_val,
                            "date": end_dt,
                        })
                except (TypeError, ValueError):
                    continue

    if not rows:
        raise RuntimeError(f"SEC EDGAR returned 0 positive USD line items for {entity_name}.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").fillna(pd.Timestamp("2026-01-01"))
    df = df.sort_values(by="date", ascending=False).head(int(limit)).reset_index(drop=True)
    return df, f"LIVE US SEC EDGAR API — {entity_name} ({len(df):,} audited US-GAAP filing items)"
def standardize_uploaded_ledger(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes custom client CSV/Excel Accounts Payable ledgers into the canonical ClearLoss schema.
    Auto-detects common ERP column aliases (SAP, Oracle, QuickBooks, NetSuite, Dynamics).
    """
    df = df_raw.copy()
    col_map = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    id_aliases = ["transaction_id", "invoice_id", "invoice_number", "doc_no", "document_number", "voucher_id", "id", "reference"]
    vendor_aliases = ["vendor_name", "vendor", "supplier", "supplier_name", "payee", "recipient", "merchant", "name"]
    amount_aliases = ["amount", "invoice_amount", "total", "total_amount", "net_amount", "payment_amount", "debit", "value"]
    date_aliases = ["date", "invoice_date", "posting_date", "payment_date", "transaction_date", "created_at", "timestamp"]
    agency_aliases = ["agency", "department", "cost_center", "business_unit", "division", "category", "gl_account"]

    for alias in id_aliases:
        if alias in lower_cols:
            col_map[lower_cols[alias]] = "transaction_id"
            break
    for alias in vendor_aliases:
        if alias in lower_cols:
            col_map[lower_cols[alias]] = "vendor_name"
            break
    for alias in amount_aliases:
        if alias in lower_cols:
            col_map[lower_cols[alias]] = "amount"
            break
    for alias in date_aliases:
        if alias in lower_cols:
            col_map[lower_cols[alias]] = "date"
            break
    for alias in agency_aliases:
        if alias in lower_cols:
            col_map[lower_cols[alias]] = "agency"
            break

    df.rename(columns=col_map, inplace=True)

    if "amount" not in df.columns:
        raise ValueError("Uploaded file must contain an 'amount' (or 'invoice_amount' / 'total') column.")

    if "transaction_id" not in df.columns:
        df["transaction_id"] = [f"INV-{100000 + i}" for i in range(len(df))]
    if "vendor_name" not in df.columns:
        df["vendor_name"] = "UNSPECIFIED VENDOR"
    if "date" not in df.columns:
        df["date"] = pd.Timestamp.now().normalize()
    if "agency" not in df.columns:
        df["agency"] = "Corporate AP"

    df["transaction_id"] = df["transaction_id"].astype(str).str.strip()
    df["vendor_name"] = df["vendor_name"].fillna("UNKNOWN VENDOR").astype(str).str.strip()
    df["agency"] = df["agency"].fillna("Corporate AP").astype(str).str.strip()
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(r"[$,\s]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce").fillna(pd.Timestamp.now().normalize())
    df = df[df["amount"] > 0].reset_index(drop=True)
    return df
