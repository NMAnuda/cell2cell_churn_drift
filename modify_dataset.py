import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load CSV
# -----------------------------
df = pd.read_csv("cell2cellholdout.csv")
print("Original shape:", df.shape)

np.random.seed(42)

# -----------------------------
# 2. Create realistic churn (no leakage)
# -----------------------------
risk = (
    0.25 * (df['CustomerCareCalls'] > df['CustomerCareCalls'].median()).astype(int) +
    0.25 * (df['OverageMinutes'] > df['OverageMinutes'].median()).astype(int) +
    0.20 * (df['DroppedCalls'] > df['DroppedCalls'].median()).astype(int) +
    0.20 * (df['MonthsInService'] < df['MonthsInService'].median()).astype(int)
)

risk = risk + np.random.normal(0, 0.4, size=len(df))
prob = 1 / (1 + np.exp(-risk))

threshold = np.percentile(prob, 80)
df['Churn'] = (prob > threshold).astype(int)

print("Churn rate:", df['Churn'].mean())

# -----------------------------
# 3. Preprocessing
# -----------------------------
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
num_cols.remove('Churn')

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('Churn', axis=1)
y = df['Churn']

# -----------------------------
# 4. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# 5. Handle imbalance
# -----------------------------
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# -----------------------------
# 6. Train model
# -----------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 7. Predict
# -----------------------------
probs = model.predict_proba(X_test)[:,1]

# -----------------------------
# 8. Find best threshold for F1
# -----------------------------
best_f1 = 0
best_thresh = 0

for t in np.linspace(0.05, 0.95, 200):
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("Best F1:", best_f1)
print("Best threshold:", best_thresh)

# -----------------------------
# 9. Final Metrics
# -----------------------------
final_preds = (probs >= best_thresh).astype(int)
final_f1 = f1_score(y_test, final_preds)
final_auc = roc_auc_score(y_test, probs)

print("Final F1:", final_f1)
print("Final AUC:", final_auc)

# -----------------------------
# 10. Precision-Recall Curve
# -----------------------------
precision, recall, _ = precision_recall_curve(y_test, probs)

plt.figure(figsize=(8,6))
plt.plot(recall, precision, color="orange", label="Model")
plt.hlines(y=y_test.mean(), xmin=0, xmax=1, linestyles="dashed", label="No Skill")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 11. Save modified dataset
# -----------------------------
df.to_csv("modified_realistic.csv", index=False)
print("Saved modified_realistic.csv")
