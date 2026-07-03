"""
Week 4 Project - System 1: Customer Churn Prediction
UCT / upskill Campus Internship - Deelipan

Target metrics: 88.4% accuracy, 0.902 ROC AUC, 89.7% recall
Includes: SMOTE balancing, fairness audit, MLflow tracking, Flask serving stub
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
df = pd.read_csv("data/customer_churn.csv")
TARGET = "churn"

# Churn is ~12% of customers (class imbalance noted in report)
print(f"Churn rate: {df[TARGET].mean():.1%}")

# ---------------------------------------------------------------------
# 2. FEATURE PREP
# ---------------------------------------------------------------------
cat_cols = df.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df, columns=cat_cols)

X = df_encoded.drop(columns=[TARGET])
y = df_encoded[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------------
# 3. CLASS IMBALANCE HANDLING
# ---------------------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ---------------------------------------------------------------------
# 4. MODEL TRAINING WITH MLFLOW TRACKING
# ---------------------------------------------------------------------
mlflow.set_experiment("customer_churn_prediction")

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=400, max_depth=14, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Optimize decision threshold based on business cost (favor recall)
    threshold = 0.35
    y_pred_adjusted = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_adjusted)
    auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred_adjusted)
    precision = precision_score(y_test, y_pred_adjusted)

    mlflow.log_params({"n_estimators": 400, "max_depth": 14, "threshold": threshold})
    mlflow.log_metrics({"accuracy": acc, "roc_auc": auc, "recall": recall, "precision": precision})
    mlflow.sklearn.log_model(model, "churn_model")

    print(f"Accuracy: {acc:.3f} | ROC AUC: {auc:.3f} | Recall: {recall:.3f} | Precision: {precision:.3f}")

# ---------------------------------------------------------------------
# 5. FAIRNESS AUDIT — check disparity across demographic groups
# ---------------------------------------------------------------------
def fairness_audit(df_test, y_true, y_pred, sensitive_col):
    if sensitive_col not in df_test.columns:
        return
    audit = pd.DataFrame({sensitive_col: df_test[sensitive_col], "actual": y_true, "pred": y_pred})
    rates = audit.groupby(sensitive_col).apply(lambda g: (g["pred"] == 1).mean())
    print(f"\nPositive prediction rate by {sensitive_col}:\n{rates}")
    disparity = rates.max() - rates.min()
    print(f"Disparity: {disparity:.1%} — {'FLAG for review' if disparity > 0.1 else 'within acceptable range'}")

if "gender" in df.columns:
    fairness_audit(df.loc[X_test.index], y_test, y_pred_adjusted, "gender")

# ---------------------------------------------------------------------
# 6. FEATURE IMPORTANCE
# ---------------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Churn Drivers:\n", importances.head(10))

# ---------------------------------------------------------------------
# 7. SIMPLE FLASK INFERENCE STUB (for deployment reference)
# ---------------------------------------------------------------------
FLASK_APP_STUB = '''
from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)
model = mlflow.sklearn.load_model("models:/churn_model/production")

@app.route("/predict", methods=["POST"])
def predict():
    data = pd.DataFrame([request.json])
    proba = model.predict_proba(data)[:, 1][0]
    return jsonify({"churn_probability": float(proba), "will_churn": bool(proba >= 0.35)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
'''
with open("src/churn_api.py", "w") as f:
    f.write(FLASK_APP_STUB)

print("\nChurn prediction pipeline complete. Flask API stub written to src/churn_api.py")
