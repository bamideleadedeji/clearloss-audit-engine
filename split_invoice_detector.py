import pandas as pd
import numpy as np


def detect_split_invoices(
    df: pd.DataFrame,
    approval_threshold: float = 10000.0,
    window_hours: int = 48,
    min_transactions: int = 2,
) -> pd.DataFrame:
    """
    Identifies structured invoice-splitting patterns designed to evade approval limits.
    Normalizes vendor names, detects 2-way, 3-way, and 4-way splits (25% to 99.99% of threshold),
    and verifies true cluster time spans.
    """
    df = df.copy()
    if df.empty:
        df["split_invoice_flag"] = False
        df["cluster_id"] = None
        df["split_risk_score"] = 0.0
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["vendor_key"] = df["vendor_name"].astype(str).str.strip().str.upper()
    df = df.sort_values(by=["vendor_key", "date"]).reset_index(drop=True)

    lower_bound = approval_threshold * 0.25
    df["is_sub_threshold"] = (df["amount"] >= lower_bound) & (df["amount"] < approval_threshold)

    df["rolling_sum_amount"] = 0.0
    df["rolling_tx_count"] = 0
    df["split_invoice_flag"] = False
    df["cluster_id"] = None

    cluster_counter = 1
    for vendor_key, group in df.groupby("vendor_key"):
        sub_group = group[group["is_sub_threshold"]].copy()
        if len(sub_group) < min_transactions:
            continue

        sub_group["time_diff"] = sub_group["date"].diff().dt.total_seconds() / 3600.0
        cluster_mask = sub_group["time_diff"].isna() | (sub_group["time_diff"] > window_hours)
        sub_group["cluster_num"] = cluster_mask.cumsum()

        for _, cluster in sub_group.groupby("cluster_num"):
            cluster_total = float(cluster["amount"].sum())
            cluster_count = len(cluster)
            span_hours = (cluster["date"].max() - cluster["date"].min()).total_seconds() / 3600.0

            if cluster_count >= min_transactions and cluster_total >= approval_threshold and span_hours <= (window_hours * 1.5):
                indices = cluster.index
                df.loc[indices, "split_invoice_flag"] = True
                df.loc[indices, "rolling_sum_amount"] = cluster_total
                df.loc[indices, "rolling_tx_count"] = cluster_count
                clean_prefix = "".join(ch for ch in vendor_key if ch.isalnum())[:4] or "VEND"
                df.loc[indices, "cluster_id"] = f"SPLIT-{clean_prefix}-{cluster_counter:03d}"
                cluster_counter += 1

    df["split_risk_score"] = np.where(df["split_invoice_flag"], 86.0, 0.0)
    df.loc[df["split_invoice_flag"], "split_risk_score"] += np.clip(
        (df["rolling_sum_amount"] - approval_threshold) / max(approval_threshold, 1.0) * 14.0,
        0.0,
        13.0,
    )
    return df


def generate_split_summary(df: pd.DataFrame, approval_threshold: float = 10000.0) -> pd.DataFrame:
    """
    Generates an executive vendor-level summary table of split-invoice risk exposure.
    """
    if df.empty or "split_invoice_flag" not in df.columns:
        return pd.DataFrame(
            columns=["vendor_name", "cluster_id", "transaction_count", "total_split_amount", "bypassed_threshold", "potential_leakage"]
        )

    flagged = df[df["split_invoice_flag"] == True]
    if flagged.empty:
        return pd.DataFrame(
            columns=["vendor_name", "cluster_id", "transaction_count", "total_split_amount", "bypassed_threshold", "potential_leakage"]
        )

    summary = (
        flagged.groupby(["vendor_name", "cluster_id"])
        .agg(
            transaction_count=("transaction_id", "count"),
            total_split_amount=("amount", "sum"),
            start_time=("date", "min"),
            end_time=("date", "max"),
        )
        .reset_index()
    )
    summary["bypassed_threshold"] = approval_threshold
    summary["potential_leakage"] = summary["total_split_amount"] - approval_threshold
    return summary.sort_values(by="total_split_amount", ascending=False)
 
