"""
Week 4 Project - System 2: House Price Forecasting
UCT / upskill Campus Internship - Deelipan

Target metrics: R2=0.923, RMSE=$42,850, MAE=$31,200
Includes: polynomial/interaction features, cross-validation with learning curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
df = pd.read_csv("data/house_prices.csv")
TARGET = "sale_price"

# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING — polynomial & interaction terms
# ---------------------------------------------------------------------
base_features = ["sqft_living", "bedrooms", "bathrooms", "lot_size", "year_built", "grade"]
base_features = [f for f in base_features if f in df.columns]

df_encoded = pd.get_dummies(df, columns=[c for c in df.select_dtypes(include="object").columns])

X_base = df_encoded[base_features].fillna(0)
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X_base)
poly_feature_names = poly.get_feature_names_out(base_features)

X = pd.DataFrame(X_poly, columns=poly_feature_names, index=df.index)
# Append remaining non-polynomial features (categoricals, etc.)
other_cols = [c for c in df_encoded.columns if c not in base_features + [TARGET]]
X = pd.concat([X, df_encoded[other_cols].fillna(0)], axis=1)
y = df_encoded[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 3. MODEL TRAINING
# ---------------------------------------------------------------------
model = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=4,
    subsample=0.8, random_state=42
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: ${rmse:,.0f} | MAE: ${mae:,.0f} | R2: {r2:.3f}")

# ---------------------------------------------------------------------
# 4. CROSS-VALIDATION WITH LEARNING CURVES (prevent overfitting)
# ---------------------------------------------------------------------
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="r2", n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(7, 5))
plt.plot(train_sizes, train_mean, "o-", label="Training R2")
plt.plot(train_sizes, val_mean, "o-", label="Validation R2")
plt.xlabel("Training examples")
plt.ylabel("R2 Score")
plt.title("Learning Curve — House Price Model")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/price_learning_curve.png")
plt.close()

print(f"\nFinal train R2: {train_mean[-1]:.3f} | Final val R2: {val_mean[-1]:.3f}")
print("Gap:", "Low (good generalization)" if abs(train_mean[-1] - val_mean[-1]) < 0.05 else "High (check overfitting)")

# ---------------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# ---------------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Price Drivers:\n", importances.head(10))

print("\nHouse price forecasting pipeline complete.")
