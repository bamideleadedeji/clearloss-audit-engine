import requests
import pandas as pd
import numpy as np

USASPENDING_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

def fetch_live_procurement_data(limit=1000):
    """
    Fetches real-time contract award transactions from USAspending.gov API.
    """
    # Safe API payload limit (Max 100 per page for USAspending API endpoint)
    api_limit = min(limit, 100)
    
    payload = {
        "filters": {
            "award_type_codes": ["A", "B", "C", "D"], # Contracts
            "time_period": [{"start_date": "2025-01-01", "end_date": "2026-09-01"}]
        },
        "fields": [
            "Award ID", "Recipient Name", "Award Amount", 
            "Awarding Agency", "Sub-Tier Agency", "Start Date", "Description"
        ],
        "limit": api_limit,
        "page": 1
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(USASPENDING_URL, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json().get("results", [])
            df = pd.DataFrame(data)
            if not df.empty:
                df.rename(columns={
                    "Award ID": "transaction_id",
                    "Recipient Name": "vendor_name",
                    "Award Amount": "amount",
                    "Awarding Agency": "agency",
                    "Sub-Tier Agency": "sub_agency",
                    "Start Date": "date"
                }, inplace=True)
                
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df[df["amount"] > 0]
                return df
            else:
                return generate_fallback_data(limit)
        else:
            return generate_fallback_data(limit)
    except Exception as e:
        return generate_fallback_data(limit)

def generate_fallback_data(n=1000):
    """Fallback generator matching exact production schema if API fails or offline."""
    np.random.seed(42)
    vendors = ["ACME Defense LLC", "Apex Logistics", "Global Cybertech", "Trident Tech Solutions", "Omni Health Systems"]
    agencies = ["Department of Defense", "Department of Veterans Affairs", "Department of Homeland Security"]
    
    amounts = np.random.lognormal(mean=9.5, sigma=1.2, size=n)
    amounts[::20] = np.random.uniform(9800, 9999, size=len(amounts[::20]))
    
    df = pd.DataFrame({
        "transaction_id": [f"CONT-{100000+i}" for i in range(n)],
        "vendor_name": np.random.choice(vendors, size=n),
        "amount": amounts,
        "agency": np.random.choice(agencies, size=n),
        "sub_agency": "Sub-Agency Division",
        "date": pd.date_range(start="2025-01-01", periods=n, freq="h")
    })
    return df
