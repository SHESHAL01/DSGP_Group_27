from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv('encoded_skills_dataset.csv')

# Features and target
X = df.drop(columns=['title'])
y = df['title']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\n" + "="*60)
print("BASELINE MODEL: GRADIENT BOOSTING")
print("="*60)

# Create model
gb_model = GradientBoostingClassifier(random_state=42)

# Train model
gb_model.fit(X_train, y_train)

# Predict
y_pred_gb = gb_model.predict(X_test)
y_proba_gb = gb_model.predict_proba(X_test)

# Accuracy
gb_accuracy = accuracy_score(y_test, y_pred_gb)
print("\nAccuracy:", gb_accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_gb))

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_gb))

# ROC-AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))
gb_roc_auc = roc_auc_score(y_test_bin, y_proba_gb, multi_class='ovr', average='macro')
print("\nMacro ROC-AUC:", gb_roc_auc)

# Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

print("\n" + "="*60)
print("BASELINE MODEL: XGBOOST")
print("="*60)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=len(np.unique(y_encoded)),
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42
)

# Train
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred_xgb))

# Convert back to original labels (for readable report)
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred_xgb)

print("\nClassification Report:\n")
print(classification_report(y_test_labels, y_pred_labels))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test_labels, y_pred_labels))

# ROC-AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

roc_auc_xgb = roc_auc_score(
    y_test_bin,
    y_prob_xgb,
    multi_class="ovr",
    average="macro"
)

print("\nMacro ROC-AUC:", roc_auc_xgb)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

print("\n" + "="*60)
print("TUNING: GRADIENT BOOSTING")
print("="*60)

gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro',
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)

best_gb = gb_grid.best_estimator_

print("Best Gradient Boosting Parameters:")
print(gb_grid.best_params_)

# Evaluate
y_pred_gb = best_gb.predict(X_test)

print("\nTuned Gradient Boosting Accuracy:",
      accuracy_score(y_test, y_pred_gb))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_gb))

from xgboost import XGBClassifier

print("\n" + "="*60)
print("TUNING: XGBOOST")
print("="*60)

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        eval_metric="mlogloss",
        random_state=42
    ),
    xgb_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro',
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_

print("Best XGBoost Parameters:")
print(xgb_grid.best_params_)

# Evaluate
y_pred_xgb = best_xgb.predict(X_test)

print("\nTuned XGBoost Accuracy:",
      accuracy_score(y_test, y_pred_xgb))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_xgb))

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np

# Binarize test labels
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

# Predict probabilities
y_score_gb = best_gb.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_gb[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2,
             label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC-AUC Curve - Tuned Gradient Boosting")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# For XGBoost (encoded labels)
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

y_score_xgb = best_xgb.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_xgb[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2,
             label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC-AUC Curve - Tuned XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import label_binarize
import numpy as np

print("\n" + "="*60)
print("OVERFITTING CHECK: TUNED GRADIENT BOOSTING")
print("="*60)

# Predictions
y_train_pred_gb = best_gb.predict(X_train)
y_test_pred_gb = best_gb.predict(X_test)

# Accuracy
train_acc_gb = accuracy_score(y_train, y_train_pred_gb)
test_acc_gb = accuracy_score(y_test, y_test_pred_gb)

print("\nTraining Accuracy:", train_acc_gb)
print("Testing Accuracy:", test_acc_gb)

# Log Loss
y_train_proba_gb = best_gb.predict_proba(X_train)
y_test_proba_gb = best_gb.predict_proba(X_test)

train_loss_gb = log_loss(y_train, y_train_proba_gb)
test_loss_gb = log_loss(y_test, y_test_proba_gb)

print("\nTraining Log Loss:", train_loss_gb)
print("Testing Log Loss:", test_loss_gb)

print("\n" + "="*60)
print("OVERFITTING CHECK: TUNED XGBOOST")
print("="*60)

# Predictions
y_train_pred_xgb = best_xgb.predict(X_train)
y_test_pred_xgb = best_xgb.predict(X_test)

# Accuracy
train_acc_xgb = accuracy_score(y_train, y_train_pred_xgb)
test_acc_xgb = accuracy_score(y_test, y_test_pred_xgb)

print("\nTraining Accuracy:", train_acc_xgb)
print("Testing Accuracy:", test_acc_xgb)

# Log Loss
y_train_proba_xgb = best_xgb.predict_proba(X_train)
y_test_proba_xgb = best_xgb.predict_proba(X_test)

train_loss_xgb = log_loss(y_train, y_train_proba_xgb)
test_loss_xgb = log_loss(y_test, y_test_proba_xgb)

print("\nTraining Log Loss:", train_loss_xgb)
print("Testing Log Loss:", test_loss_xgb)

import matplotlib.pyplot as plt

models = ['Gradient Boosting', 'XGBoost']
train_scores = [train_acc_gb, train_acc_xgb]
test_scores = [test_acc_gb, test_acc_xgb]

x = np.arange(len(models))

plt.figure(figsize=(8,6))
plt.bar(x - 0.2, train_scores, width=0.4, label='Training Accuracy')
plt.bar(x + 0.2, test_scores, width=0.4, label='Testing Accuracy')

plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.legend()
plt.grid(axis='y')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

train_acc = []
test_acc = []
train_loss = []
test_loss = []

n_estimators_range = range(10, 301, 20)

for n in n_estimators_range:
    model = GradientBoostingClassifier(
        n_estimators=n,
        learning_rate=best_gb.learning_rate,
        max_depth=best_gb.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Accuracy
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

    # Loss
    train_loss.append(log_loss(y_train, model.predict_proba(X_train)))
    test_loss.append(log_loss(y_test, model.predict_proba(X_test)))

# Plot Accuracy Curve
plt.figure(figsize=(8, 6))
plt.plot(n_estimators_range, train_acc, label="Training Accuracy")
plt.plot(n_estimators_range, test_acc, label="Testing Accuracy")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.title("Gradient Boosting Accuracy Curve")
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(n_estimators_range, train_loss, label="Training Loss")
plt.plot(n_estimators_range, test_loss, label="Testing Loss")
plt.xlabel("Number of Estimators")
plt.ylabel("Log Loss")
plt.title("Gradient Boosting Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

train_acc = []
test_acc = []
train_loss = []
test_loss = []

n_estimators_range = range(10, 301, 20)

for n in n_estimators_range:
    model = XGBClassifier(
        n_estimators=n,
        learning_rate=best_gb.learning_rate,
        max_depth=best_gb.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Accuracy
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

    # Loss
    train_loss.append(log_loss(y_train, model.predict_proba(X_train)))
    test_loss.append(log_loss(y_test, model.predict_proba(X_test)))

# Plot Accuracy Curve
plt.figure(figsize=(8, 6))
plt.plot(n_estimators_range, train_acc, label="Training Accuracy")
plt.plot(n_estimators_range, test_acc, label="Testing Accuracy")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.title("XGBoost Accuracy Curve")
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(n_estimators_range, train_loss, label="Training Loss")
plt.plot(n_estimators_range, test_loss, label="Testing Loss")
plt.xlabel("Number of Estimators")
plt.ylabel("Log Loss")
plt.title("XGBoost Loss Curve")
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score

print("\n" + "=" * 60)
print("SENSITIVITY CHECK: TUNED GRADIENT BOOSTING")
print("=" * 60)

# Predictions
y_pred_gb = best_gb.predict(X_test)

# Confusion Matrix
cm_gb = confusion_matrix(y_test, y_pred_gb)
classes = np.unique(y_test)

# Sensitivity (Recall)
sensitivity_gb = recall_score(y_test, y_pred_gb, average=None)

# PPV (Precision)
ppv_gb = precision_score(y_test, y_pred_gb, average=None)

# NPV calculation (manual)
npv_gb = []

for i in range(len(classes)):
    TP = cm_gb[i, i]
    FN = np.sum(cm_gb[i, :]) - TP
    FP = np.sum(cm_gb[:, i]) - TP
    TN = np.sum(cm_gb) - (TP + FN + FP)

    npv = TN / (TN + FN)
    npv_gb.append(npv)

npv_gb = np.array(npv_gb)

# Print metrics
for i, cls in enumerate(classes):
    print(f"\nClass: {cls}")
    print("Sensitivity (Recall):", sensitivity_gb[i])
    print("PPV (Precision):", ppv_gb[i])
    print("NPV:", npv_gb[i])

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, sensitivity_gb, width, label='Sensitivity')
plt.bar(x, ppv_gb, width, label='PPV')
plt.bar(x + width, npv_gb, width, label='NPV')

plt.xticks(x, classes, rotation=45)
plt.ylabel("Score")
plt.title("Sensitivity Analysis - Tuned Gradient Boosting")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("SENSITIVITY CHECK: TUNED XGBOOST")
print("=" * 60)

# Predictions
y_pred_xgb = best_xgb.predict(X_test)

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

# Sensitivity
sensitivity_xgb = recall_score(y_test, y_pred_xgb, average=None)

# PPV
ppv_xgb = precision_score(y_test, y_pred_xgb, average=None)

# NPV calculation
npv_xgb = []

for i in range(len(classes)):
    TP = cm_xgb[i, i]
    FN = np.sum(cm_xgb[i, :]) - TP
    FP = np.sum(cm_xgb[:, i]) - TP
    TN = np.sum(cm_xgb) - (TP + FN + FP)

    npv = TN / (TN + FN)
    npv_xgb.append(npv)

npv_xgb = np.array(npv_xgb)

# Print metrics
for i, cls in enumerate(classes):
    print(f"\nClass: {cls}")
    print("Sensitivity (Recall):", sensitivity_xgb[i])
    print("PPV (Precision):", ppv_xgb[i])
    print("NPV:", npv_xgb[i])

plt.figure(figsize=(10,6))
plt.bar(x - width, sensitivity_xgb, width, label='Sensitivity')
plt.bar(x, ppv_xgb, width, label='PPV')
plt.bar(x + width, npv_xgb, width, label='NPV')

plt.xticks(x, classes, rotation=45)
plt.ylabel("Score")
plt.title("Sensitivity Analysis - Tuned XGBoost")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score

print("\n" + "=" * 60)
print("YOUDEN INDEX: TUNED GRADIENT BOOSTING")
print("=" * 60)

# Predictions
y_pred_gb = best_gb.predict(X_test)

# Confusion Matrix
cm_gb = confusion_matrix(y_test, y_pred_gb)
classes = np.unique(y_test)

youden_gb = []

for i in range(len(classes)):
    TP = cm_gb[i, i]
    FN = np.sum(cm_gb[i, :]) - TP
    FP = np.sum(cm_gb[:, i]) - TP
    TN = np.sum(cm_gb) - (TP + FN + FP)

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    youden = sensitivity + specificity - 1
    youden_gb.append(youden)

    print(f"\nClass: {classes[i]}")
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Youden Index:", youden)

youden_gb = np.array(youden_gb)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.bar(classes, youden_gb)
plt.xticks(rotation=45)
plt.ylabel("Youden Index")
plt.title("Youden Index per Class - Tuned Gradient Boosting")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("YOUDEN INDEX: TUNED XGBOOST")
print("=" * 60)

# Predictions
y_pred_xgb = best_xgb.predict(X_test)

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

youden_xgb = []

for i in range(len(classes)):
    TP = cm_xgb[i, i]
    FN = np.sum(cm_xgb[i, :]) - TP
    FP = np.sum(cm_xgb[:, i]) - TP
    TN = np.sum(cm_xgb) - (TP + FN + FP)

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    youden = sensitivity + specificity - 1
    youden_xgb.append(youden)

    print(f"\nClass: {classes[i]}")
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Youden Index:", youden)

youden_xgb = np.array(youden_xgb)

plt.figure(figsize=(8,6))
plt.bar(classes, youden_xgb)
plt.xticks(rotation=45)
plt.ylabel("Youden Index")
plt.title("Youden Index per Class - Tuned XGBoost")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

import joblib
import os

# create folder first
os.makedirs("saved_models", exist_ok=True)

metrics = {
    "XGBoost": {
        "accuracy": accuracy_score(y_test, y_pred_xgb)
    },
    "Gradient Boosting": {
        "accuracy": accuracy_score(y_test, y_pred_gb)
    }
}

joblib.dump(metrics, "saved_models/model_metrics.pkl")
joblib.dump(best_xgb, "saved_models/best_xgb_model.pkl")
joblib.dump(best_gb, "saved_models/best_gb_model.pkl")

print("All models saved successfully!")

