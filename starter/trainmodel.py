from sklearn.metrics import average_precision_score, roc_auc_score

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Read the CSV and reconstruct X, targets, and weights
df = pd.read_csv("xy_dataset.csv")

target_cols = ["y_click", "y_buy", "sample_weights"]
y_click = df["y_click"].to_numpy()
y_buy = df["y_buy"].to_numpy()
sample_weights = df["sample_weights"].to_numpy()
X = df.drop(columns=target_cols)

# CSV loses pandas 'category' dtypes; convert string/object columns back for LightGBM
for col in X.select_dtypes(include=["object", "string"]).columns:
    X[col] = X[col].astype("category")

# 2. Apply extra weight to interactions that resulted in a purchase
BUY_WEIGHT_MULTIPLIER = 5.0
combined_weights = sample_weights * np.where(y_buy == 1, BUY_WEIGHT_MULTIPLIER, 1.0)
y_target = np.maximum(y_click, y_buy)

# 3. Train/validation split and model training
X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    X,
    y_target,
    combined_weights,
    test_size=0.2,
    random_state=42,
)

model = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=1000,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    sample_weight=w_train,
    eval_set=[(X_val, y_val)],
    eval_sample_weight=[w_val],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

scores = model.predict_proba(X_val)[:, 1]

model.booster_.save_model("lgbm_click_buy_model.txt")

# 1. Overall validation metrics (runs immediately with your existing y_val, w_val, scores)
print("=== Overall Validation Metrics ===")
print(f"Weighted ROC-AUC:   {roc_auc_score(y_val, scores, sample_weight=w_val):.4f}")
print(f"Unweighted ROC-AUC: {roc_auc_score(y_val, scores):.4f}")
print(f"Weighted PR-AUC:    {average_precision_score(y_val, scores, sample_weight=w_val):.4f}")
print(f"Unweighted PR-AUC:  {average_precision_score(y_val, scores):.4f}")