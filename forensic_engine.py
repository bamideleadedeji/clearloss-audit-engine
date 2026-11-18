import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats import chisquare


def analyze_benfords_law(df: pd.DataFrame) -> dict:
    """
    Evaluates first-digit AND first-two-digit distributions against Benford's Law,
    including Mark Nigrini's Mean Absolute Deviation (MAD) conformity classification.
    """
    if df is None or df.empty or "amount" not in df.columns:
        return {}

    amounts = pd.to_numeric(df["amount"], errors="coerce").dropna().values
    positive_amounts = [abs(float(v)) for v in amounts if float(v) > 0]

    first_digits = []
    first_two_digits = []

    for val in positive_amounts:
        digits_only = f"{val:.6f}".replace(".", "").lstrip("0")
        if len(digits_only) >= 1 and digits_only[0].isdigit() and digits_only[0] != "0":
            first_digits.append(int(digits_only[0]))
        if len(digits_only) >= 2 and digits_only[:2].isdigit():
            d2 = int(digits_only[:2])
            if 10 <= d2 <= 99:
                first_two_digits.append(d2)

    if not first_digits:
        return {}

    n = len(first_digits)
    counts = pd.Series(first_digits).value_counts().reindex(range(1, 10), fill_value=0)
    observed_freq = counts / n
    benford_expected = np.log10(1 + 1 / np.arange(1, 10))

    chi_stat, p_value = chisquare(counts, f_exp=benford_expected * n)

    # Nigrini Mean Absolute Deviation (MAD) Conformity Standard
    mad = float(np.mean(np.abs(observed_freq.values - benford_expected)))
    if mad <= 0.006:
        mad_status = "Close Conformity (Natural)"
    elif mad <= 0.012:
        mad_status = "Acceptable Conformity"
    elif mad <= 0.015:
        mad_status = "Marginally Acceptable Conformity"
    else:
        mad_status = "Nonconformity — High Manipulation Risk"

    # First-Two-Digit (10-99) spike detection (catches 98/99 threshold clustering)
    two_digit_spikes = []
    if len(first_two_digits) >= 30:
        n2 = len(first_two_digits)
        c2 = pd.Series(first_two_digits).value_counts().reindex(range(10, 100), fill_value=0)
        obs2 = c2 / n2
        exp2 = np.log10(1 + 1 / np.arange(10, 100))
        z2 = (obs2.values - exp2) / np.sqrt((exp2 * (1 - exp2)) / n2)
        for idx, d_pair in enumerate(range(10, 100)):
            if z2[idx] >= 2.58 and c2.loc[d_pair] >= 3:
                two_digit_spikes.append({
                    "Leading_Two_Digits": d_pair,
                    "Observed_Count": int(c2.loc[d_pair]),
                    "Observed_Pct": round(float(obs2.loc[d_pair]) * 100, 2),
                    "Expected_Pct": round(float(exp2[idx]) * 100, 2),
                    "Z_Score": round(float(z2[idx]), 2),
                })

    return {
        "digit": list(range(1, 10)),
        "observed": observed_freq.values.tolist(),
        "expected": benford_expected.tolist(),
        "chi_stat": float(chi_stat),
        "p_value": float(p_value),
        "mad": mad,
        "mad_status": mad_status,
        "sample_size": n,
        "two_digit_spikes": two_digit_spikes,
    }


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines Isolation Forest ML, Vendor Z-Score outliers, Exact Duplicate Invoice checks,
    and Round-Dollar threshold billing flags into an explainable 0-100 risk score.
    """
    df = df.copy()
    if df.empty:
        df["risk_score"] = 0.0
        df["flagged_reason"] = "Normal"
        df["is_duplicate"] = False
        df["is_round_dollar"] = False
        return df

    df["vendor_key"] = df["vendor_name"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["log_amount"] = np.log1p(df["amount"])

    vendor_mean = df.groupby("vendor_key")["amount"].transform("mean").fillna(0.0)
    vendor_std = df.groupby("vendor_key")["amount"].transform("std").fillna(0.0)
    df["vendor_zscore"] = np.where(vendor_std > 0, (df["amount"] - vendor_mean) / vendor_std, 0.0)

    # 1. Isolation Forest Multivariate Outlier Detection
    if len(df) >= 10:
        contamination_rate = min(0.08, max(2.0 / len(df), 0.02))
        model = IsolationForest(contamination=contamination_rate, random_state=42)
        df["anomaly_score"] = model.fit_predict(df[["log_amount", "vendor_zscore"]])
    else:
        df["anomaly_score"] = 1

    # 2. Exact Duplicate Invoice Detection (Same Vendor + Same Amount within 7 Days)
    df = df.sort_values(by=["vendor_key", "amount", "date"]).reset_index(drop=True)
    same_vendor_amt = (df["vendor_key"] == df["vendor_key"].shift(1)) & (np.isclose(df["amount"], df["amount"].shift(1), atol=0.01))
    days_apart = (df["date"] - df["date"].shift(1)).dt.total_seconds().abs() / 86400.0
    df["is_duplicate"] = same_vendor_amt & (days_apart <= 7.0)

    # 3. Suspicious Round-Dollar Billing ($5,000+ exact multiples of $1,000)
    df["is_round_dollar"] = (df["amount"] >= 5000.0) & (np.isclose(df["amount"] % 1000.0, 0.0, atol=0.01))

    # Base Risk Score from ML & Z-Score
    base_score = np.where(
        df["anomaly_score"] == -1,
        np.clip(np.abs(df["vendor_zscore"]) * 20.0 + 62.0, 62.0, 96.0),
        np.clip(np.abs(df["vendor_zscore"]) * 12.0, 5.0, 48.0),
    )
    df["risk_score"] = base_score
    df.loc[df["is_round_dollar"], "risk_score"] = np.maximum(df["risk_score"], 68.0)
    df.loc[df["is_duplicate"], "risk_score"] = np.maximum(df["risk_score"], 90.0)

    # Explainable Audit Flag Reasons
    reasons = []
    for _, r in df.iterrows():
        tags = []
        if r["is_duplicate"]:
            tags.append("Duplicate Payment (<7d)")
        if r["vendor_zscore"] > 3.0:
            tags.append(f"Vendor Z-Score Outlier ({r['vendor_zscore']:.1f}σ)")
        elif r["anomaly_score"] == -1:
            tags.append("Isolation Forest Multivariate Outlier")
        if r["is_round_dollar"]:
            tags.append("Round-Dollar Billing")
        reasons.append(" | ".join(tags) if tags else "Normal")

    df["flagged_reason"] = reasons
    return df
