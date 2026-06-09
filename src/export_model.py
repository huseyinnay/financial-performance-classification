import pandas as pd
import numpy as np
import json
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load original data and split as in the notebook to recreate exactly the same training set
financial_data = pd.read_csv("../data/raw/financial_data.csv")
financial_data["Company_ID"] = (financial_data["Total Assets"] * 1000 + financial_data["Total Liabilities"]).astype(int)

from zlib import crc32
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

train_set, test_set = split_data_with_id_hash(financial_data, 0.2, "Company_ID")

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

train_set = add_features(train_set)

numerical_features = [
    "Revenue", "Net Income", "Total Assets", "Equity", "Market Capitalization", 
    "EBITDA_Margin", "ROA", "ROE", "CashFlow_to_Debt", "DSCR", "FCF", "Inventory_Turnover", "DSO", "P_B_Ratio"
]
log_features = ["Revenue", "Total Assets", "Equity", "Market Capitalization"]

train_set_scaled = train_set.copy()
train_set_scaled[log_features] = np.log1p(train_set_scaled[log_features])

scaler = StandardScaler()
train_set_scaled[numerical_features] = scaler.fit_transform(train_set_scaled[numerical_features])

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(train_set_scaled[numerical_features])

# Save preprocessing parameters
preprocessing_params = {
    "numerical_features": numerical_features,
    "log_features": log_features,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "imputer_statistics": imputer.statistics_.tolist()
}

with open("../models/preprocessing.json", "w") as f:
    json.dump(preprocessing_params, f, indent=4)

print("Preprocessing parameters saved to preprocessing.json")

# Export SVM to ONNX
best_model = joblib.load("../models/svm_financial_model.pkl")
initial_type = [('float_input', FloatTensorType([None, len(numerical_features)]))]
onx = convert_sklearn(best_model, initial_types=initial_type)

with open("../models/financial_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("Model exported to financial_model.onnx")
