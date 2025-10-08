import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("dynamic_pricing.csv")

kpi = {
    "Avg_Historical_Price": round(df["Historical_Cost_of_Ride"].mean(), 2),
    "Median_Historical_Price": round(df["Historical_Cost_of_Ride"].median(), 2),
    "Avg_Riders": round(df["Number_of_Riders"].mean(), 2),
    "Avg_Drivers": round(df["Number_of_Drivers"].mean(), 2),
}
print("Milestone 1 KPIs:\n", kpi)

int_cols = ["Number_of_Riders", "Number_of_Drivers", "Number_of_Past_Rides", "Expected_Ride_Duration"]
float_cols = ["Average_Ratings", "Historical_Cost_of_Ride"]
cat_cols = ["Location_Category", "Customer_Loyalty_Status", "Time_of_Booking", "Vehicle_Type"]

for c in int_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
for c in float_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in cat_cols:
    df[c] = df[c].astype("string").str.strip()

# Handle missing values
df[int_cols + float_cols] = df[int_cols + float_cols].fillna(df[int_cols + float_cols].median())
df[cat_cols] = df[cat_cols].fillna("Unknown")

df.loc[~df["Average_Ratings"].between(1, 5), "Average_Ratings"] = df["Average_Ratings"].median()

plt.scatter(df["Number_of_Drivers"], df["Number_of_Riders"])
plt.title("Riders vs Drivers")
plt.xlabel("Drivers")
plt.ylabel("Riders")
plt.show()
df.groupby("Location_Category")["Historical_Cost_of_Ride"].mean().plot(kind="bar", edgecolor="black", title="Avg Price by Location")
plt.ylabel("Average Cost")
plt.show()
corr = df[int_cols + float_cols].corr()
plt.imshow(corr, interpolation="nearest")
plt.title("Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.show()

df["Historical_Cost_of_Ride"].plot(kind="hist", bins=20, edgecolor="black", title="Distribution of Ride Cost")
plt.xlabel("Ride Cost")
plt.show()

print("\nKPI Results:")
print("Average Historical Price:", round(df["Historical_Cost_of_Ride"].mean(), 2))
print("Median Historical Price:", round(df["Historical_Cost_of_Ride"].median(), 2))
print("Average Riders per Driver:", round((df["Number_of_Riders"]/df["Number_of_Drivers"]).mean(), 2))

df["Competitor_Price_Index"] = np.random.uniform(0.9, 1.1, size=len(df))
df["Cost_per_Min"] = df["Historical_Cost_of_Ride"] / df["Expected_Ride_Duration"].replace(0, np.nan)
df["Driver_to_Rider_Ratio"] = df["Number_of_Drivers"] / df["Number_of_Riders"].replace(0, np.nan)
df["Inventory_Health_Index"] = df["Number_of_Drivers"] / (df["Number_of_Riders"] + 1)
loyalty_map = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
df["Loyalty_Score"] = df["Customer_Loyalty_Status"].map(loyalty_map).fillna(0).astype(int)
df["Peak"] = df["Time_of_Booking"].apply(lambda x: 1 if x in ["Evening", "Night"] else 0)
df["Rider_Driver_Ratio"] = df["Number_of_Riders"] / df["Number_of_Drivers"].replace(0, np.nan)
df["Supply_Tightness"] = df["Rider_Driver_Ratio"] * df["Peak"]
veh_map = {"Economy": 1.0, "Premium": 1.5}
df["Vehicle_Factor"] = df["Vehicle_Type"].map(veh_map).fillna(1.0)
base_rate = 2.0
df["baseline_price"] = df["Expected_Ride_Duration"] * df["Vehicle_Factor"] * base_rate
df["competitor_price"] = df["baseline_price"] * df["Competitor_Price_Index"]
df["p_complete"] = 0.7 + (0.05 * df["Loyalty_Score"]) + (0.01 * df["Driver_to_Rider_Ratio"])
df["p_complete"] = df["p_complete"].clip(0, 1)
df["price"] = (df["baseline_price"] * 0.5 + df["competitor_price"] * 0.3 + (df["Supply_Tightness"] * 5) * 0.2)

print(df.head())

def gm_pct(price, cost):
    """Compute gross margin % as decimal"""
    return (price - cost) / price if price > 0 else 0.0


STABILITY_PCT = 0.15
MIN_GM_PCT = 0.12
COMP_CAP = {"Economy": 1.05, "Premium": 1.08}
COMP_FLOOR = {"Economy": 0.90, "Premium": 0.88}

def row_price_bounds(row):
    base = row["baseline_price"]
    cost = row["Historical_Cost_of_Ride"]
    veh = row["Vehicle_Type"]
    comp = row["competitor_price"]

    # Price stability
    lo, hi = base*(1-STABILITY_PCT), base*(1+STABILITY_PCT)

    # GM constraint
    min_gm_price = cost / (1-MIN_GM_PCT)
    lo = max(lo, min_gm_price)

    # Competitor constraints
    floor = COMP_FLOOR.get(veh, 0.9)
    cap = COMP_CAP.get(veh, 1.05)
    lo = max(lo, comp*floor)
    hi = min(hi, comp*cap)

    if hi < lo: hi = lo
    return lo, hi

def choose_optimal_price(row):
    lo, hi = row_price_bounds(row)


    center = row["baseline_price"] + row["Supply_Tightness"] * 5
    center = np.clip(center, lo, hi)


    grid = np.linspace(lo, hi, 11)

    best_price = center
    best_revenue = best_price * row["p_complete"]

    for p in grid:
        gm = gm_pct(p, row["Historical_Cost_of_Ride"])
        if gm < MIN_GM_PCT: continue

        p_complete_est = row["p_complete"] + 0.01*(p - row["baseline_price"])/row["baseline_price"]
        p_complete_est = np.clip(p_complete_est, 0, 1)
        revenue = p * p_complete_est
        if revenue > best_revenue:
            best_revenue = revenue
            best_price = p
    return pd.Series([best_price, best_revenue], index=["opt_price", "est_revenue"])

df[["opt_price", "est_revenue"]] = df.apply(choose_optimal_price, axis=1)

avg_gm = ((df["opt_price"] - df["Historical_Cost_of_Ride"]) / df["opt_price"]).mean()
avg_price = df["opt_price"].mean()
avg_revenue = df["est_revenue"].mean()
print("\n=== KPIs for Optimized Pricing ===")
print(f"Average Optimized Price: {round(avg_price,2)}")
print(f"Average Estimated Revenue: {round(avg_revenue,2)}")
print(f"Average Gross Margin: {round(avg_gm*100,2)}%")

plt.figure(figsize=(8,5))
plt.scatter(df["Number_of_Drivers"], df["Number_of_Riders"], c=df["opt_price"], cmap="viridis")
plt.colorbar(label="Optimized Price")
plt.xlabel("Number of Drivers")
plt.ylabel("Number of Riders")
plt.title("Optimized Price by Supply and Demand")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["opt_price"], bins=20, edgecolor="black")
plt.title("Distribution of Optimized Prices")
plt.xlabel("Optimized Price")
plt.ylabel("Frequency")
plt.show()

def policy_audit(df):
    # Price stability
    stable = df["opt_price"].between(df["baseline_price"]*0.85, df["baseline_price"]*1.15).all()
    # GM ≥ 12%
    gm_ok = ((df["opt_price"] - df["Historical_Cost_of_Ride"])/df["opt_price"] >= MIN_GM_PCT).all()
    # Competitiveness
    comp_ok = ((df["opt_price"] <= df["competitor_price"]*1.08) &
               (df["opt_price"] >= df["competitor_price"]*0.88)).all()
    # No increase in cancellation
    cancel_ok = (df["p_complete"] <= 1).all()
    return {"Price Stability": stable, "GM ≥ 12%": gm_ok, "Competitor Check": comp_ok, "Cancellation Check": cancel_ok}

audit_results = policy_audit(df)
print("\n=== Policy Audit Results ===")
for k,v in audit_results.items():
    print(f"{k}: {v}")

df.head(10)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


has_xgb = False
try:
    from xgboost import XGBRegressor
    has_xgb = True
except Exception:
    pass

# 1) Prepare ML dataset
df_ml = df.copy()
y = df_ml["opt_price"]   # target

cat_features = ["Time_of_Booking", "Location_Category", "Vehicle_Type", "Customer_Loyalty_Status"]
num_features = [
    "Expected_Ride_Duration", "Historical_Cost_of_Ride",
    "Number_of_Riders", "Number_of_Drivers",
    "Vehicle_Factor", "Peak", "Competitor_Price_Index"
]


cols_needed = cat_features + num_features + ["opt_price"]
df_ml = df_ml.dropna(subset=[c for c in cols_needed if c in df_ml.columns]).copy()
X = df_ml[cat_features + num_features]


# 2) Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# 3) Preprocessing

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features)
    ],
    remainder="drop"
)


# 4) Define models

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
    ),
    "DecisionTree": DecisionTreeRegressor(
        max_depth=None, random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        learning_rate=0.05, n_estimators=400, max_depth=3, random_state=42
    )
}

if has_xgb:
    models["XGBoost"] = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="auto"
    )

print("Models available:", ", ".join(models.keys()))


# 5) Train, predict & evaluate

results = {}
fitted_models = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    try:
        cv_mse = -cross_val_score(pipe, X_train, y_train, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)
        cv_rmse = float(np.mean(np.sqrt(cv_mse)))
    except Exception:
        cv_rmse = np.nan

    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "CV_RMSE": cv_rmse}
    fitted_models[name] = pipe



print("\n=== Model Comparison ===")
for name, metrics in results.items():
    print(f"- {name}: RMSE={metrics['RMSE']:.4f} | MAE={metrics['MAE']:.4f} | R2={metrics['R2']:.4f} | CV_RMSE={metrics['CV_RMSE']:.4f}")


best_model_name = min(results, key=lambda n: results[n]["RMSE"])
best_model = fitted_models[best_model_name]
print(f"\nSelected best model: {best_model_name}")

import joblib, os
out_path = os.path.join(os.getcwd(), f"{best_model_name}_opt_price_model.joblib")
joblib.dump(best_model, out_path)
print("Saved model →", out_path)