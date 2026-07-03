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
print("MUMBAI\n")
df = pd.read_csv(r'src\Mumbai House Prices.csv')
df = df[["bhk", "type", "locality", "region", "area",
         "price", "price_unit", "status", "age"]]

def convert_price(row):
    if row["price_unit"] == "L":    
        return row["price"] * 1e5
    elif row["price_unit"] == "Cr": 
        return row["price"] * 1e7
    else: return np.nan

df["price_in_inr"]   = df.apply(convert_price, axis=1)
df = df.dropna(subset=["price_in_inr", "area"])
df = df[(df["area"] > 200) & (df["area"] < 5000)]
df["price_per_sqft"] = df["price_in_inr"] / df["area"]
df = df[(df["price_per_sqft"] > 2000) & (df["price_per_sqft"] < 200000)]

print(f"Rows after cleaning : {len(df)}")

os.makedirs("rag_dfs", exist_ok=True)
df.to_csv("rag_dfs/mumbai_cleaned_dataset.csv", index=False)
print("Saved → rag_dfs/mumbai_cleaned_dataset.csv")

X = df[["bhk", "area", "type", "region", "status", "age"]]
y = df["price_per_sqft"]

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

#Random Forest
print("\n Random Forest")
mae, r2, pipe = evaluate("Random Forest", make_pipeline(
    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
))
results["Random Forest"] = {"mae": mae, "r2": r2, "pipeline": pipe}

#XGBoost
print("\n XGBoost")
pipe = make_pipeline(XGBRegressor(
    n_estimators=600,
    max_depth=10,
    learning_rate=0.09,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
))
mae, r2, pipe = evaluate("XGBoost", pipe)
results["XGBoost"] = {"mae": mae, "r2": r2, "pipeline": pipe}

#Summary
for name, res in results.items():
    print(f"  {name:<30}  R²: {res['r2']:.4f}   MAE: ₹{res['mae']:.2f}/sqft")

#joblib.dump(results["Linear Regression"]["pipeline"], "models/mumbai_price_model.pkl")
joblib.dump(results["Random Forest"]["pipeline"], "models/mumbai_price_model.pkl")
#joblib.dump(results["XGBoost"]["pipeline"], "models/mumbai_price_model.pkl")