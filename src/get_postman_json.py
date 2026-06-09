import pandas as pd
import json

df = pd.read_csv("../data/processed/test_set.csv")

def add_features(data):
    data = data.copy()
    data["EBITDA_Margin"] = data["EBITDA"] / data["Revenue"]
    data["ROA"] = data["Net Income"] / data["Total Assets"]
    data["ROE"] = data["Net Income"] / data["Equity"]
    data["CashFlow_to_Debt"] = data["Operating Cash Flow"] / data["Total Debt"]
    data["DSCR"] = data["EBITDA"] / data["Interest Expense"]
    data["FCF"] = data["Operating Cash Flow"] - data["Capital Expenditures"]
    data["Inventory_Turnover"] = data["Cost of Goods Sold"] / (data["Total Assets"] - data["Equity"])
    data["DSO"] = data["Accounts Receivable Turnover"] / (data["Revenue"] / 365)
    data["P_B_Ratio"] = data["Market Capitalization"] / data["Equity"]
    return data

df = add_features(df)
features = [
    "Revenue", "Net Income", "Total Assets", "Equity", "Market Capitalization", 
    "EBITDA_Margin", "ROA", "ROE", "CashFlow_to_Debt", "DSCR", "FCF", "Inventory_Turnover", "DSO", "P_B_Ratio"
]

for i in range(5):
    row = df.iloc[i][features].to_dict()
    print(f"--- Satır {i+1} ---")
    print(json.dumps(row, indent=4))
