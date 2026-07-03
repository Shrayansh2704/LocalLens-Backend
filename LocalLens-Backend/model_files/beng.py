import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

print("Bengaluru\n")
df = pd.read_csv(r'src\Bengaluru_House_Data.csv')

df["bhk"] = df["size"].str.extract(r'(\d+)').astype(float)

def convert_sqft(x):
    try:
        if isinstance(x, str):
            if "-" in x:
                vals = x.split("-")
                return (float(vals[0]) + float(vals[1])) / 2
            if "Sq. Meter" in x:
                return float(x.replace("Sq. Meter", "").strip()) * 10.7639
            if "Sq. Yard" in x:
                return float(x.replace("Sq. Yard", "").strip()) * 9
        return float(x)
    except:
        return np.nan

df["area"] = df["total_sqft"].apply(convert_sqft)
df["type"] = df["area_type"]
df["locality"] = df["location"]
df["region"] = df["location"]
df["price_unit"] = "L"
df["status"] = df["availability"].apply(
    lambda x: "Ready" if "Ready" in str(x) else "Under Construction"
)
df["age"] = df["availability"].apply(
    lambda x: "New" if "Ready" not in str(x) else "Unknown"
)

df = df[["bhk", "type", "locality", "region", "area",
         "price", "price_unit", "status", "age"]]

def convert_price(row):
    if row["price_unit"] == "L":    
        return row["price"] * 1e5
    elif row["price_unit"] == "Cr": 
        return row["price"] * 1e7
    else: return np.nan

df["price_in_inr"] = df.apply(convert_price, axis=1)
df = df.dropna(subset=["price_in_inr", "area"])
df = df[(df["area"] > 200) & (df["area"] < 5000)]
df["price_per_sqft"] = df["price_in_inr"] / df["area"]
df = df[(df["price_per_sqft"] > 2000) & (df["price_per_sqft"] < 200000)]

print(f"Rows after cleaning : {len(df)}")

os.makedirs("rag_dfs", exist_ok=True)
df.to_csv("rag_dfs/bengaluru_cleaned_dataset.csv", index=False)
print("Saved → rag_dfs/bengaluru_cleaned_dataset.csv")

# Features
X = df[["bhk", "area", "type", "region", "status", "age"]]
y = df["price_per_sqft"]

X = X.dropna()
y = y[X.index]

cat_features = ["type", "region", "status", "age"]
num_features = ["bhk", "area"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.18, random_state=42
)

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", "passthrough", num_features),
])

def make_pipeline(reg):
    return Pipeline([("preprocessor", preprocessor), ("regressor", reg)])

def evaluate(name, pipe):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    print(f"  {name:<30}  R²: {r2:.4f}   MAE: ₹{mae:.2f}/sqft")
    return mae, r2, pipe

results = {}

#Linear Regression
print("\n Linear Regression")
mae, r2, pipe = evaluate("Linear Regression", make_pipeline(LinearRegression()))
results["Linear Regression"] = {"mae": mae, "r2": r2, "pipeline": pipe}

# Random Forest
print("\n Random Forest")
mae, r2, pipe = evaluate("Random Forest", make_pipeline(
    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
))
results["Random Forest"] = {"mae": mae, "r2": r2, "pipeline": pipe}

#XGBoost
print("\n XGBoost")
pipe = make_pipeline(XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.09,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
))
mae, r2, pipe = evaluate("XGBoost", pipe)
results["XGBoost"] = {"mae": mae, "r2": r2, "pipeline": pipe}
#--------------------------------------------------------------------------------- 
for name, res in results.items():
    print(f"  {name:<30}  {res['r2']:>8.4f}   ₹{res['mae']:.2f}")
#--------------------------------------------------------------------------------

#joblib.dump(results["Linear Regression"]["pipeline"], "models/bengaluru_price_model.pkl")
#joblib.dump(results["Random Forest"]["pipeline"], "models/bengaluru_price_model.pkl")
joblib.dump(results["XGBoost"]["pipeline"], "models/bengaluru_price_model.pkl")
