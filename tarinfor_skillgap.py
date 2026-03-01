# ============================================================
# BASELINE MODEL TRAINING - Logistic Regression & Random Forest
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("encoded_skills_dataset.csv")

X = df.drop(columns=["title"])
y = df["title"]

# ============================================================
# 2. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

classes = np.unique(y)

# ============================================================
# 3. BASELINE LOGISTIC REGRESSION
# ============================================================

print("\n" + "="*60)
print("BASELINE MODEL: LOGISTIC REGRESSION")
print("="*60)

log_model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
)

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred_log))

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_log))

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_log))

# ROC-AUC (One-vs-Rest Macro)
y_test_bin = label_binarize(y_test, classes=classes)
y_prob_log = log_model.predict_proba(X_test)

roc_auc_log = roc_auc_score(
    y_test_bin,
    y_prob_log,
    multi_class="ovr",
    average="macro"
)

print("\nMacro ROC-AUC:", roc_auc_log)

# Cross-validation
cv_scores_log = cross_val_score(
    log_model,
    X,
    y,
    cv=5,
    scoring="f1_macro"
)

print("\n5-Fold CV F1-Macro Mean:", cv_scores_log.mean())

# ============================================================
# 4. BASELINE RANDOM FOREST
# ============================================================

print("\n" + "="*60)
print("BASELINE MODEL: RANDOM FOREST")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred_rf))

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_rf))

# ROC-AUC
y_prob_rf = rf_model.predict_proba(X_test)

roc_auc_rf = roc_auc_score(
    y_test_bin,
    y_prob_rf,
    multi_class="ovr",
    average="macro"
)

print("\nMacro ROC-AUC:", roc_auc_rf)

# Cross-validation
cv_scores_rf = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="f1_macro"
)

print("\n5-Fold CV F1-Macro Mean:", cv_scores_rf.mean())

# ============================================================
# 5. Model Comparison Summary
# ============================================================

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

print(f"Logistic Regression ROC-AUC: {roc_auc_log:.4f}")
print(f"Random Forest ROC-AUC:       {roc_auc_rf:.4f}")

print(f"Logistic Regression CV F1:   {cv_scores_log.mean():.4f}")
print(f"Random Forest CV F1:         {cv_scores_rf.mean():.4f}")

# ============================================================
# FINE TUNING MODELS
# ============================================================

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

print("\n" + "="*60)
print("FINE TUNING: LOGISTIC REGRESSION")
print("="*60)

# -----------------------------
# Logistic Regression Grid
# -----------------------------
log_param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "newton-cg"],
    "max_iter": [500, 1000, 2000]
}

log_grid = GridSearchCV(
    LogisticRegression(),
    log_param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

log_grid.fit(X_train, y_train)
best_log = log_grid.best_estimator_

print("Best Parameters (LR):", log_grid.best_params_)

# Evaluation
y_pred_log = best_log.predict(X_test)
y_prob_log = best_log.predict_proba(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred_log))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_log))

roc_auc_log = roc_auc_score(
    label_binarize(y_test, classes=best_log.classes_),
    y_prob_log,
    multi_class="ovr"
)

print("Macro ROC-AUC:", roc_auc_log)


# ============================================================
# RANDOM FOREST TUNING
# ============================================================

print("\n" + "="*60)
print("FINE TUNING: RANDOM FOREST")
print("="*60)

rf_param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [None, 20, 40],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced"]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

print("Best Parameters (RF):", rf_grid.best_params_)

# Evaluation
y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_rf))

roc_auc_rf = roc_auc_score(
    label_binarize(y_test, classes=best_rf.classes_),
    y_prob_rf,
    multi_class="ovr"
)

print("Macro ROC-AUC:", roc_auc_rf)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "="*60)
print("TUNED MODEL COMPARISON")
print("="*60)

print(f"Tuned Logistic Regression ROC-AUC: {roc_auc_log:.4f}")
print(f"Tuned Random Forest ROC-AUC:       {roc_auc_rf:.4f}")
print(f"Tuned Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(f"Tuned Random Forest Accuracy:       {accuracy_score(y_test, y_pred_rf):.4f}")


# ============================================================
# COMBINED ROC CURVE (ONE GRAPH)
# ============================================================

print("\nPlotting Combined ROC-AUC Curve...\n")

# Binarize test labels
classes = best_log.classes_
y_test_bin = label_binarize(y_test, classes=classes)

# Compute micro-average ROC for LR
fpr_log, tpr_log, _ = roc_curve(
    y_test_bin.ravel(),
    y_prob_log.ravel()
)

# Compute micro-average ROC for RF
fpr_rf, tpr_rf, _ = roc_curve(
    y_test_bin.ravel(),
    y_prob_rf.ravel()
)

plt.figure(figsize=(8,6))

plt.plot(fpr_log, tpr_log,
         label=f"Logistic Regression (AUC = {roc_auc_log:.3f})")

plt.plot(fpr_rf, tpr_rf,
         label=f"Random Forest (AUC = {roc_auc_rf:.3f})")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (Tuned Models)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
