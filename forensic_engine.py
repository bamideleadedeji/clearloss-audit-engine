import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats import chisquare

def analyze_benfords_law(df):
    """
    Evaluates first-digit distribution of transaction amounts against Benford's Law.
    """
    amounts = df["amount"].dropna().values
    first_digits = [int(str(abs(val)).replace('.', '').lstrip('0')[0]) for val in amounts if val > 0]
    
    if len(first_digits) == 0:
        return {}, 0.0
        
    counts = pd.Series(first_digits).value_counts().reindex(range(1, 10), fill_value=0)
    observed_freq = counts / len(first_digits)
    
    # Standard Benford's Law distribution values
    benford_expected = np.log10(1 + 1 / np.arange(1, 10))
    
    # Chi-square goodness of fit test
    chi_stat, p_value = chisquare(counts, f_exp=benford_expected * len(first_digits))
    
    benford_summary = {
        "digit": list(range(1, 10)),
        "observed": observed_freq.values.tolist(),
        "expected": benford_expected.tolist(),
        "chi_stat": float(chi_stat),
        "p_value": float(p_value)
    }
    return benford_summary

def run_anomaly_detection(df):
    """
    Applies Isolation Forest & Z-Score heuristics to score transaction risk.
    """
    df = df.copy()
    
    if df.empty:
        df["risk_score"] = 0
        df["flagged_reason"] = "Normal"
        return df
    
    # Feature Engineering
    df["log_amount"] = np.log1p(df["amount"])
    
    # Fixed groupby transforms for single statistics
    vendor_mean = df.groupby("vendor_name")["amount"].transform("mean").fillna(0)
    vendor_std = df.groupby("vendor_name")["amount"].transform("std").fillna(0)
    
    df["vendor_zscore"] = (df["amount"] - vendor_mean) / (vendor_std + 1e-5)
    
    # Isolation Forest Model
    contamination_rate = min(0.05, max(1.0 / len(df), 0.01)) # Dynamic safe contamination
    model = IsolationForest(contamination=contamination_rate, random_state=42)
    features = df[["log_amount", "vendor_zscore"]]
    df["anomaly_score"] = model.fit_predict(features) # -1 for anomaly, 1 for normal
    
    # Calculate Risk Score (0 - 100)
    df["risk_score"] = np.where(
        df["anomaly_score"] == -1, 
        np.clip(df["vendor_zscore"] * 25 + 50, 60, 99), 
        np.clip(df["vendor_zscore"] * 10, 5, 40)
    )
    
    # Calculate flag status
    df["flagged_reason"] = "Normal"
    df.loc[df["risk_score"] > 60, "flagged_reason"] = "High Amount Anomaly"
    df.loc[df["vendor_zscore"] > 3, "flagged_reason"] = "Vendor Z-Score Outlier (>3σ)"
    
    return df
