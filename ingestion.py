import requests
import pandas as pd
import numpy as np

USASPENDING_URL = "https://usaspending.gov"


def fetch_live_procurement_data(limit: int = 300) -> tuple[pd.DataFrame, str]:
    """
    Fetches real-time federal contract award transactions from USAspending.gov API
    using multi-page pagination (100 records per page).
    Returns (DataFrame, source_status_label).
    """
    headers = {"Content-Type": "application/json", "User-Agent": "ClearLoss-Forensic-Audit/2.0"}
    all_rows = []
    pages_needed = max(1, (int(limit) + 99) // 100)

    try:
        for page_num in range(1, pages_needed + 1):
            batch_size = min(100, int(limit) - len(all_rows))
            if batch_size <= 0:
                break
            payload = {
                "filters": {
                    "award_type_codes": ["A", "B", "C", "D"],
                    "time_period": [{"start_date": "2025-01-01", "end_date": "2026-11-18"}],
                },
                "fields": [
                    "Award ID",
                    "Recipient Name",
                    "Award Amount",
                    "Awarding Agency",
                    "Sub-Tier Agency",
                    "Start Date",
                    "Description",
                ],
                "limit": batch_size,
                "page": page_num,
                "sort": "Award Amount",
                "order": "desc",
            }
            response = requests.post(USASPENDING_URL, json=payload, headers=headers, timeout=15)
            if response.status_code != 200:
                break
            results = response.json().get("results", [])
            if not results:
                break
            all_rows.extend(results)
            if len(results) < batch_size:
                break

        if all_rows:
            df = pd.DataFrame(all_rows)
            df.rename(
                columns={
                    "Award ID": "transaction_id",
                    "Recipient Name": "vendor_name",
                    "Award Amount": "amount",
                    "Awarding Agency": "agency",
                    "Sub-Tier Agency": "sub_agency",
                    "Start Date": "date",
                    "Description": "description",
                },
                inplace=True,
            )
            df["transaction_id"] = df["transaction_id"].fillna("UNASSIGNED-ID").astype(str)
            df["vendor_name"] = df["vendor_name"].fillna("UNKNOWN VENDOR").astype(str).str.strip()
            df["agency"] = df.get("agency", pd.Series("Federal Agency", index=df.index)).fillna("Federal Agency").astype(str)
            df["sub_agency"] = df.get("sub_agency", pd.Series("General Division", index=df.index)).fillna("General Division").astype(str)
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
            df["date"] = pd.to_datetime(df["date"], errors="coerce").fillna(pd.Timestamp("2026-01-01"))
            df = df[df["amount"] > 0].reset_index(drop=True)
            if not df.empty:
                return df, f"LIVE USAspending.gov Federal API ({len(df):,} verified contracts)"
    except Exception:
        pass

    return generate_fallback_data(limit), "OFFLINE DEMO DATASET (USAspending.gov API unreachable — simulated ledger)"


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
    if "sub_agency" not in df.columns:
        df["sub_agency"] = "General Ledger"

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


def generate_fallback_data(n: int = 300) -> pd.DataFrame:
    """Offline demonstration ledger matching production schema when API is unavailable."""
    np.random.seed(42)
    vendors = [
        "ACME Defense Systems LLC",
        "Apex Logistics Group",
        "Global Cybertech Inc.",
        "Trident Federal Solutions",
        "Omni Health Partners",
        "Vanguard Infrastructure Corp",
        "Meridian IT Consulting",
    ]
    agencies = [
        "Department of Defense",
        "Department of Veterans Affairs",
        "Department of Homeland Security",
        "Department of Health and Human Services",
    ]

    amounts = np.round(np.random.lognormal(mean=9.3, sigma=1.15, size=n), 2)
    # Inject realistic split-invoice evasion clusters just below $10,000
    for idx in range(10, min(n - 3, 160), 35):
        amounts[idx] = 9850.00
        amounts[idx + 1] = 9920.00
        amounts[idx + 2] = 9640.00

    dates = pd.date_range(start="2026-01-05", periods=n, freq="6h")
    df = pd.DataFrame(
        {
            "transaction_id": [f"CONT-{100000 + i}" for i in range(n)],
            "vendor_name": np.random.choice(vendors, size=n),
            "amount": amounts,
            "agency": np.random.choice(agencies, size=n),
            "sub_agency": "Procurement Operations",
            "date": dates,
        }
    )
    # Align split-invoice clusters to single vendors within 24 hours
    for idx in range(10, min(n - 3, 160), 35):
        target_vendor = df.loc[idx, "vendor_name"]
        base_dt = df.loc[idx, "date"]
        df.loc[idx + 1, "vendor_name"] = target_vendor
        df.loc[idx + 1, "date"] = base_dt + pd.Timedelta(hours=5)
        df.loc[idx + 2, "vendor_name"] = target_vendor
        df.loc[idx + 2, "date"] = base_dt + pd.Timedelta(hours=19)

    return df
