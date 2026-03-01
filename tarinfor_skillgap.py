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

