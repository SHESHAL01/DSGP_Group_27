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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure()

    # Training Accuracy Curve
    plt.plot(
        train_sizes,
        train_mean,
        marker='o',
        linewidth=2,
        label="Training Accuracy"
    )

    # Testing Accuracy Curve
    plt.plot(
        train_sizes,
        test_mean,
        marker='s',
        linewidth=2,
        label="Validation Accuracy"
    )

    plt.title(title + " - Learning Curve (Accuracy)")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot for both models
plot_learning_curve(best_log, X, y, "Logistic Regression (Tuned)")
plot_learning_curve(best_rf, X, y, "Random Forest (Tuned)")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import log_loss


def plot_loss_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring='neg_log_loss',  # sklearn returns NEGATIVE log loss
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    # Convert negative log loss to positive
    train_loss = -np.mean(train_scores, axis=1)
    test_loss = -np.mean(test_scores, axis=1)

    plt.figure()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def compute_metrics_per_class(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    results = []

    for i in range(len(class_names)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        # Safe division to avoid ZeroDivisionError
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        ppv = TP / (TP + FP) if (TP + FP) != 0 else 0
        npv = TN / (TN + FN) if (TN + FN) != 0 else 0
        youden = sensitivity + specificity - 1

        results.append([class_names[i], sensitivity, ppv, npv, youden])

    return pd.DataFrame(
        results,
        columns=["Class", "Sensitivity", "PPV", "NPV", "Youden"]
    )
# ----- DEFINE METRICS -----
class_names = best_log.classes_

lr_metrics = compute_metrics_per_class(best_log, X_test, y_test, class_names)
rf_metrics = compute_metrics_per_class(best_rf, X_test, y_test, class_names)

print(lr_metrics)
print(rf_metrics)

# ----- GROUPED BAR FUNCTION -----
def plot_metrics_grouped(df, title):
    classes = df["Class"]
    x = np.arange(len(classes))
    width = 0.25   # bar width

    plt.figure(figsize=(10,6))

    plt.bar(x - width, df["Sensitivity"], width, label="Sensitivity")
    plt.bar(x, df["PPV"], width, label="PPV")
    plt.bar(x + width, df["NPV"], width, label="NPV")

    plt.xticks(x, classes, rotation=45)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----- CALL PLOTS -----
plot_metrics_grouped(rf_metrics, "Sensitivity / PPV / NPV - Random Forest")
plot_metrics_grouped(lr_metrics, "Sensitivity / PPV / NPV - Logistic Regression")

def plot_youden_comparison(lr_df, rf_df):
    classes = lr_df["Class"]
    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(10,6))

    plt.bar(x - width/2, lr_df["Youden"], width, label="Logistic Regression")
    plt.bar(x + width/2, rf_df["Youden"], width, label="Random Forest")

    plt.xticks(x, classes, rotation=45)
    plt.ylabel("Youden Index")
    plt.title("Youden Index per Class: LR vs RF")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_youden_comparison(lr_metrics, rf_metrics)