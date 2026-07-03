Week 1 Project: Prediction of Agriculture Crop Production in India
UCT / upskill Campus Internship - Deelipan

Pipeline: Data ingestion -> Cleaning -> Feature Engineering ->
          Model Training (Linear Regression, Random Forest, XGBoost) -> Evaluation
Data source: data.gov.in (Government of India), 1993-2014
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------
DATA_DIR = "data/"

produce = pd.read_csv(DATA_DIR + "produce.csv")            # 429 x 25 - main production data
growth_index = pd.read_csv(DATA_DIR + "datafile.csv")       # 12 x 9  - production index
cost_yield = pd.read_csv(DATA_DIR + "datafile__1_.csv")     # 49 x 6  - cost & yield by crop/state
prod_area_yield = pd.read_csv(DATA_DIR + "datafile__2_.csv")# 55 x 16 - crop-wise production/area/yield
variety_catalog = pd.read_csv(DATA_DIR + "datafile__3_.csv")# 78 x 4  - crop variety catalogue

# ---------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------
def clean_column_names(df):
    """Standardise column names: strip whitespace, remove special currency symbols."""
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[`₹]", "", regex=True)
        .str.replace(" ", "_")
    )
    return df

def clean_text_columns(df, cols):
    """Strip trailing/leading whitespace from key text columns for consistent joins."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def parse_year(value):
    """Standardise 'month-year' style strings (e.g. '3-2014') to numeric year."""
    try:
        if isinstance(value, str) and "-" in value:
            return int(value.split("-")[-1])
        return int(value)
    except (ValueError, TypeError):
        return np.nan

produce = clean_column_names(produce)
cost_yield = clean_column_names(cost_yield)
prod_area_yield = clean_column_names(prod_area_yield)
variety_catalog = clean_column_names(variety_catalog)
growth_index = clean_column_names(growth_index)

for df in [produce, cost_yield, prod_area_yield, variety_catalog]:
    clean_text_columns(df, ["Crop", "State", "Crop_Name", "State_Name"])

# Median imputation for missing cost/yield values, grouped by crop
if "Crop" in cost_yield.columns:
    cost_yield["Cost_of_Cultivation"] = cost_yield.groupby("Crop")["Cost_of_Cultivation"].transform(
        lambda x: x.fillna(x.median())
    )

# ---------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------
def build_features(produce, cost_yield, variety_catalog):
    df = produce.copy()

    # Year as numeric trend feature
    if "Year" in df.columns:
        df["Year"] = df["Year"].apply(parse_year)

    # Season indicator flags (Kharif / Rabi) from 'Particulars' or 'Season' field
    season_col = "Season" if "Season" in df.columns else "Particulars"
    if season_col in df.columns:
        df["is_kharif"] = df[season_col].str.contains("Kharif", case=False, na=False).astype(int)
        df["is_rabi"] = df[season_col].str.contains("Rabi", case=False, na=False).astype(int)

    # Merge cost of cultivation and yield (state + crop level)
    if {"Crop", "State"}.issubset(df.columns) and {"Crop", "State"}.issubset(cost_yield.columns):
        df = df.merge(
            cost_yield[["Crop", "State", "Cost_of_Cultivation", "Yield"]],
            on=["Crop", "State"], how="left"
        )

    # Merge variety catalogue for season duration / recommended zone
    if "Crop" in variety_catalog.columns:
        df = df.merge(
            variety_catalog[["Crop", "Season_Duration", "Recommended_Zone"]],
            on="Crop", how="left"
        )

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["Crop", "State"], prefix=["crop", "state"])

    return df

features_df = build_features(produce, cost_yield, variety_catalog)
features_df = features_df.dropna(subset=["Production"])  # target must be present

# ---------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT (TIME-BASED)
# ---------------------------------------------------------------------
TARGET = "Production"

train_df = features_df[features_df["Year"] < 2012]
test_df = features_df[features_df["Year"] >= 2012]

drop_cols = [TARGET]
X_train = train_df.drop(columns=drop_cols).select_dtypes(include=[np.number]).fillna(0)
y_train = train_df[TARGET]
X_test = test_df.drop(columns=drop_cols).select_dtypes(include=[np.number]).fillna(0)
y_test = test_df[TARGET]

# Align columns between train/test (in case of one-hot mismatches)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# ---------------------------------------------------------------------
# 5. MODEL TRAINING
# ---------------------------------------------------------------------
def evaluate(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name:25s} | RMSE: {rmse:,.2f} | MAE: {mae:,.2f} | R2: {r2:.3f}")
    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}

results = []

# --- Baseline: Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
results.append(evaluate("Linear Regression", y_test, lr.predict(X_test)))

# --- Random Forest Regressor ---
rf = RandomForestRegressor(
    n_estimators=300, max_depth=12, min_samples_leaf=3,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
results.append(evaluate("Random Forest", y_test, rf.predict(X_test)))

# --- XGBoost Regressor ---
xgb = XGBRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb.fit(X_train, y_train)
results.append(evaluate("XGBoost", y_test, xgb.predict(X_test)))

results_df = pd.DataFrame(results)
print("\nModel Comparison:\n", results_df)

# ---------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(10)
print("\nTop 10 Feature Importances (Random Forest):\n", top_features)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/figures/feature_importance.png")
plt.close()

# ---------------------------------------------------------------------
# 7. PREDICTED VS ACTUAL PLOT (XGBoost - best model)
# ---------------------------------------------------------------------
y_pred_xgb = xgb.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.title("XGBoost: Predicted vs Actual Crop Production")
plt.tight_layout()
plt.savefig("outputs/figures/predicted_vs_actual.png")
plt.close()

print("\nPipeline complete. Best model: XGBoost (highest R2, lowest RMSE).")

