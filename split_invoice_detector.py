import pandas as pd
import numpy as np

def detect_split_invoices(
    df: pd.DataFrame, 
    approval_threshold: float = 10000.0, 
    window_hours: int = 48, 
    min_transactions: int = 2
) -> pd.DataFrame:
    """
    Identifies structured invoice splitting patterns.
    
    Parameters:
    - df: Dataframe containing columns ['transaction_id', 'vendor_name', 'amount', 'date', 'agency']
    - approval_threshold: The single-transaction limit that requires elevated approval (e.g., $10,000)
    - window_hours: The time window (in hours) to group suspicious transactions
    - min_transactions: Minimum number of transactions in the window required to trigger a flag
    
    Returns:
    - Annotated DataFrame with split-invoice risk metrics and flag reasons.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['vendor_name', 'date'])
    
    # 1. Isolate sub-threshold transactions (e.g., transactions between 50% and 99.9% of threshold)
    lower_bound = approval_threshold * 0.50
    df['is_sub_threshold'] = (df['amount'] >= lower_bound) & (df['amount'] < approval_threshold)
    
    # 2. Apply sliding window per vendor to calculate cumulative metrics
    df['rolling_sum_amount'] = 0.0
    df['rolling_tx_count'] = 0
    df['split_invoice_flag'] = False
    df['cluster_id'] = None
    
    flagged_indices = []
    
    # Process each vendor independently
    for vendor, group in df.groupby('vendor_name'):
        sub_group = group[group['is_sub_threshold']].copy()
        
        if len(sub_group) < min_transactions:
            continue
            
        # Group transactions falling within the window_hours threshold
        sub_group['time_diff'] = sub_group['date'].diff().dt.total_seconds() / 3600.0
        
        # Assign cluster IDs when time difference exceeds window_hours
        cluster_mask = (sub_group['time_diff'].isna()) | (sub_group['time_diff'] > window_hours)
        sub_group['cluster_num'] = cluster_mask.cumsum()
        
        # Evaluate each cluster
        for cluster_id, cluster in sub_group.groupby('cluster_num'):
            cluster_total = cluster['amount'].sum()
            cluster_count = len(cluster)
            
            # FLAG CONDITIONS:
            # - More than min_transactions in window
            # - Combined total exceeds the approval threshold
            if cluster_count >= min_transactions and cluster_total >= approval_threshold:
                indices = cluster.index
                df.loc[indices, 'split_invoice_flag'] = True
                df.loc[indices, 'rolling_sum_amount'] = cluster_total
                df.loc[indices, 'rolling_tx_count'] = cluster_count
                df.loc[indices, 'cluster_id'] = f"SPLIT-{vendor[:4].upper()}-{cluster_id:03d}"
    
    # 3. Update risk scores and reason codes
    df['split_risk_score'] = np.where(df['split_invoice_flag'], 85.0, 0.0)
    
    # Increase risk score if the combined total is significantly higher than threshold
    df.loc[df['split_invoice_flag'], 'split_risk_score'] += np.clip(
        (df['rolling_sum_amount'] - approval_threshold) / approval_threshold * 15, 0, 14
    )
    
    return df


def generate_split_summary(df: pd.DataFrame, approval_threshold: float = 10000.0) -> pd.DataFrame:
    """
    Generates a vendor-level summary table of split-invoice risk exposure.
    """
    flagged = df[df['split_invoice_flag'] == True]
    
    if flagged.empty:
        return pd.DataFrame(columns=['vendor_name', 'cluster_id', 'transaction_count', 'total_split_amount', 'bypassed_threshold'])
        
    summary = flagged.groupby(['vendor_name', 'cluster_id']).agg(
        transaction_count=('transaction_id', 'count'),
        total_split_amount=('amount', 'sum'),
        start_time=('date', 'min'),
        end_time=('date', 'max')
    ).reset_index()
    
    summary['bypassed_threshold'] = approval_threshold
    summary['potential_leakage'] = summary['total_split_amount'] - approval_threshold
    
    return summary.sort_values(by='total_split_amount', ascending=False)


if __name__ == "__main__":
    # Unit Test with synthetic split-invoice patterns
    sample_data = pd.DataFrame({
        'transaction_id': ['TX101', 'TX102', 'TX103', 'TX104', 'TX105'],
        'vendor_name': ['Apex Tech Solutions', 'Apex Tech Solutions', 'Apex Tech Solutions', 'Global Logistics', 'Global Logistics'],
        'amount': [9850.00, 9920.00, 9500.00, 15000.00, 2500.00], # Apex splits 3 transactions under $10k
        'date': [
            '2026-09-10 09:00:00', 
            '2026-09-10 14:30:00', 
            '2026-09-11 11:00:00', 
            '2026-09-10 10:00:00',
            '2026-09-12 10:00:00'
        ],
        'agency': ['Dept of Defense', 'Dept of Defense', 'Dept of Defense', 'Veterans Affairs', 'Veterans Affairs']
    })
    
    annotated_df = detect_split_invoices(sample_data, approval_threshold=10000.0, window_hours=48)
    summary_df = generate_split_summary(annotated_df, approval_threshold=10000.0)
    
    print("--- ANNOTATED TRANSACTION LEDGER ---")
    print(annotated_df[['transaction_id', 'vendor_name', 'amount', 'split_invoice_flag', 'cluster_id', 'split_risk_score']])
    print("\n--- EXECUTIVE VENDOR RISK SUMMARY ---")
    print(summary_df)
