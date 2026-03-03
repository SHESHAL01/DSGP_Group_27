import pandas as pd
import numpy as np
import xgboost
from pip._internal.commands import install

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


# Load dataset
df = pd.read_csv('encoded_skills_dataset.csv')

# Features and target
X = df.drop(columns=['title'])
y = df['title']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Logistic Regression model
log_model = LogisticRegression(
    max_iter=1000,
)

# Train
log_model.fit(X_train, y_train)

# Predict
y_pred = log_model.predict(X_test)

# Evaluation
print("Logistic Regression Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('encoded_skills_dataset.csv')

# Features and target
X = df.drop(columns=['title'])
y = df['title']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# Train
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Evaluation
print("Random Forest Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('./encoded_skills_dataset.csv')

X = df.drop(columns=['title'])
y = df['title']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'newton-cg'],
    'max_iter': [500, 1000, 2000]
}

log_grid = GridSearchCV(
    LogisticRegression(),
    log_param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

log_grid.fit(X_train, y_train)
best_log = log_grid.best_estimator_

print("Best Logistic Regression Parameters:")
print(log_grid.best_params_)

y_pred_log = best_log.predict(X_test)
print("Tuned Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('./encoded_skills_dataset.csv')

X = df.drop(columns=['title'])
y = df['title']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression Grid Search
log_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'newton-cg'],
    'max_iter': [500, 1000, 2000]
}

log_grid = GridSearchCV(
    LogisticRegression(),
    log_param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

log_grid.fit(X_train, y_train)
best_log = log_grid.best_estimator_

print("Best Logistic Regression Parameters:")
print(log_grid.best_params_)

y_pred_log = best_log.predict(X_test)
print("Tuned Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Random Forest Grid Search
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

print("Best Random Forest Parameters:")
print(rf_grid.best_params_)

y_pred_rf = best_rf.predict(X_test)
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

# Binarize the output for multiclass ROC-AUC
classes = y.unique()
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

# --- Logistic Regression ROC-AUC ---
y_score_log = best_log.predict_proba(X_test)

fpr_log = dict()
tpr_log = dict()
roc_auc_log = dict()

for i in range(n_classes):
    fpr_log[i], tpr_log[i], _ = roc_curve(y_test_bin[:, i], y_score_log[:, i])
    roc_auc_log[i] = auc(fpr_log[i], tpr_log[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr_log[i], tpr_log[i], lw=2,
             label=f"Class {classes[i]} (AUC = {roc_auc_log[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.title("ROC-AUC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# --- Random Forest ROC-AUC ---
y_score_rf = best_rf.predict_proba(X_test)

fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()

for i in range(n_classes):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], y_score_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr_rf[i], tpr_rf[i], lw=2,
             label=f"Class {classes[i]} (AUC = {roc_auc_rf[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.title("ROC-AUC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

print("\n" + "="*60)
print("BASELINE MODEL: GRADIENT BOOSTING")
print("="*60)

gb_model = GradientBoostingClassifier(random_state=42)

# Train
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

# ROC-AUC (Macro)
y_test_bin = label_binarize(y_test, classes=np.unique(y))
gb_roc_auc = roc_auc_score(y_test_bin, y_proba_gb, multi_class='ovr', average='macro')
print("\nMacro ROC-AUC:", gb_roc_auc)


from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

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
    cv=5,
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
    cv=5,
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



# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
#
# train_acc = []
# test_acc = []
# max_iter_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#
# for iters in max_iter_list:
#     lr = LogisticRegression(max_iter=iters, solver='lbfgs')
#     lr.fit(X_train, y_train)
#
#     # Track training accuracy
#     y_train_pred = lr.predict(X_train)
#     train_acc.append(accuracy_score(y_train, y_train_pred))
#
#     # Track testing accuracy
#     y_test_pred = lr.predict(X_test)
#     test_acc.append(accuracy_score(y_test, y_test_pred))
#
# plt.figure(figsize=(8, 6))
# plt.plot(max_iter_list, train_acc, label="Train Accuracy", marker='o')
# plt.plot(max_iter_list, test_acc, label="Test Accuracy", marker='o')
# plt.xlabel("Max Iterations")
# plt.ylabel("Accuracy")
# plt.title("Logistic Regression: Train vs Test Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# train_acc_rf = []
# test_acc_rf = []
#
# rf = RandomForestClassifier(n_estimators=10, warm_start=True, random_state=42, class_weight='balanced')
#
# for n_trees in range(10, 310, 10):
#     rf.n_estimators = n_trees
#     rf.fit(X_train, y_train)
#
#     # Track training accuracy
#     y_train_pred_rf = rf.predict(X_train)
#     train_acc_rf.append(accuracy_score(y_train, y_train_pred_rf))
#
#     # Track testing accuracy
#     y_test_pred_rf = rf.predict(X_test)
#     test_acc_rf.append(accuracy_score(y_test, y_test_pred_rf))
#
# plt.figure(figsize=(8, 6))
# plt.plot(range(10, 310, 10), train_acc_rf, label="Train Accuracy", marker='o')
# plt.plot(range(10, 310, 10), test_acc_rf, label="Test Accuracy", marker='o')
# plt.xlabel("Number of Trees")
# plt.ylabel("Accuracy")
# plt.title("Random Forest: Train vs Test Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# from sklearn.metrics import log_loss
#
# train_loss = []
# test_loss = []
# max_iter_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#
# for iters in max_iter_list:
#     lr = LogisticRegression(max_iter=iters, solver='lbfgs')
#     lr.fit(X_train, y_train)
#
#     # Predicted probabilities
#     y_train_prob = lr.predict_proba(X_train)
#     y_test_prob = lr.predict_proba(X_test)
#
#     # Compute log loss
#     train_loss.append(log_loss(y_train, y_train_prob))
#     test_loss.append(log_loss(y_test, y_test_prob))
#
# plt.figure(figsize=(8, 6))
# plt.plot(max_iter_list, train_loss, label="Train Loss", marker='o')
# plt.plot(max_iter_list, test_loss, label="Test Loss", marker='o')
# plt.xlabel("Max Iterations")
# plt.ylabel("Log Loss")
# plt.title("Logistic Regression: Train vs Test Loss")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# train_loss_rf = []
# test_loss_rf = []
#
# rf = RandomForestClassifier(n_estimators=10, warm_start=True, random_state=42, class_weight='balanced')
#
# for n_trees in range(10, 310, 10):
#     rf.n_estimators = n_trees
#     rf.fit(X_train, y_train)
#
#     # Use 1 - accuracy as "loss"
#     train_loss_rf.append(1 - accuracy_score(y_train, rf.predict(X_train)))
#     test_loss_rf.append(1 - accuracy_score(y_test, rf.predict(X_test)))
#
# plt.figure(figsize=(8, 6))
# plt.plot(range(10, 310, 10), train_loss_rf, label="Train Loss", marker='o')
# plt.plot(range(10, 310, 10), test_loss_rf, label="Test Loss", marker='o')
# plt.xlabel("Number of Trees")
# plt.ylabel("Error (1 - Accuracy)")
# plt.title("Random Forest: Train vs Test Error")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# #sensitivity check with PPV & NPV
# from sklearn.metrics import confusion_matrix
# import pandas as pd
# import numpy as np
#
# def multiclass_sensitivity_ppv_npv(y_true, y_pred, class_labels):
#     results = []
#
#     for cls in class_labels:
#         # One-vs-Rest transformation
#         y_true_bin = (y_true == cls).astype(int)
#         y_pred_bin = (y_pred == cls).astype(int)
#
#         tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
#
#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#         ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
#         npv = tn / (tn + fn) if (tn + fn) > 0 else 0
#
#         results.append({
#             "Class": cls,
#             "Sensitivity (Recall)": sensitivity,
#             "Specificity": specificity,
#             "PPV (Precision)": ppv,
#             "NPV": npv
#         })
#
#     return pd.DataFrame(results)
#
# #Logistic Regression
# print("\nSensitivity / PPV / NPV - Logistic Regression\n")
# metrics_log = multiclass_sensitivity_ppv_npv(
#     y_test,
#     y_pred_log,
#     class_labels=classes
# )
# print(metrics_log)
#
# print("\nMacro Averages (LR):")
# print(metrics_log.mean(numeric_only=True))
#
# #Random Forest
# print("\nSensitivity / PPV / NPV - Random Forest\n")
# metrics_rf = multiclass_sensitivity_ppv_npv(
#     y_test,
#     y_pred_rf,
#     class_labels=classes
# )
# print(metrics_rf)
#
# print("\nMacro Averages (RF):")
# print(metrics_rf.mean(numeric_only=True))
#
# #sensitivity check plots
#
# import matplotlib.pyplot as plt
#
# metrics_log.set_index("Class")[["Sensitivity (Recall)", "PPV (Precision)", "NPV"]].plot(
#     kind="bar",
#     figsize=(10, 6)
# )
#
# plt.title("Sensitivity / PPV / NPV - Logistic Regression")
# plt.ylabel("Score")
# plt.ylim(0, 1.05)
# plt.xticks(rotation=30)
# plt.grid(axis="y")
# plt.tight_layout()
# plt.show()
#
# metrics_rf.set_index("Class")[["Sensitivity (Recall)", "PPV (Precision)", "NPV"]].plot(
#     kind="bar",
#     figsize=(10, 6)
# )
#
# plt.title("Sensitivity / PPV / NPV - Random Forest")
# plt.ylabel("Score")
# plt.ylim(0, 1.05)
# plt.xticks(rotation=30)
# plt.grid(axis="y")
# plt.tight_layout()
# plt.show()
#
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
#
# thresholds = np.linspace(0, 1, 50)
#
# # Get probabilities
# prob_log = best_log.predict_proba(X_test)
#
# for target_class in best_log.classes_:
#
#     print(f"\nThreshold Analysis - Logistic Regression: {target_class}")
#
#     class_index = list(best_log.classes_).index(target_class)
#     prob_target = prob_log[:, class_index]
#
#     y_true_binary = (y_test == target_class).astype(int)
#
#     sens = []
#     spec = []
#
#     for t in thresholds:
#         y_pred_binary = (prob_target >= t).astype(int)
#
#         tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
#
#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#
#         sens.append(sensitivity)
#         spec.append(specificity)
#
#     plt.figure(figsize=(7,5))
#     plt.plot(thresholds, sens, label="Sensitivity")
#     plt.plot(thresholds, spec, label="Specificity")
#     plt.title(f"LR Threshold Tradeoff - {target_class}")
#     plt.xlabel("Decision Threshold")
#     plt.ylabel("Score")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
# prob_rf = best_rf.predict_proba(X_test)
#
# for target_class in best_rf.classes_:
#
#     print(f"\nThreshold Analysis - Random Forest: {target_class}")
#
#     class_index = list(best_rf.classes_).index(target_class)
#     prob_target = prob_rf[:, class_index]
#
#     y_true_binary = (y_test == target_class).astype(int)
#
#     sens = []
#     spec = []
#
#     for t in thresholds:
#         y_pred_binary = (prob_target >= t).astype(int)
#
#         tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
#
#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#
#         sens.append(sensitivity)
#         spec.append(specificity)
#
#     plt.figure(figsize=(7,5))
#     plt.plot(thresholds, sens, label="Sensitivity")
#     plt.plot(thresholds, spec, label="Specificity")
#     plt.title(f"RF Threshold Tradeoff - {target_class}")
#     plt.xlabel("Decision Threshold")
#     plt.ylabel("Score")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
#
# thresholds = np.linspace(0, 1, 50)
# prob_log = best_log.predict_proba(X_test)
#
# plt.figure(figsize=(8,6))
#
# for target_class in best_log.classes_:
#
#     class_index = list(best_log.classes_).index(target_class)
#     prob_target = prob_log[:, class_index]
#     y_true_binary = (y_test == target_class).astype(int)
#
#     sens = []
#     spec = []
#
#     for t in thresholds:
#         y_pred_binary = (prob_target >= t).astype(int)
#         tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
#
#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#
#         sens.append(sensitivity)
#         spec.append(specificity)
#
#     balanced_score = np.array(sens) + np.array(spec)
#     best_idx = np.argmax(balanced_score)
#
#     plt.scatter(sens[best_idx], spec[best_idx], label=target_class)
#
# plt.xlabel("Sensitivity")
# plt.ylabel("Specificity")
# plt.title("Logistic Regression - Optimal Sensitivity vs Specificity per Class")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# prob_rf = best_rf.predict_proba(X_test)
#
# plt.figure(figsize=(8,6))
#
# for target_class in best_rf.classes_:
#
#     class_index = list(best_rf.classes_).index(target_class)
#     prob_target = prob_rf[:, class_index]
#     y_true_binary = (y_test == target_class).astype(int)
#
#     sens = []
#     spec = []
#
#     for t in thresholds:
#         y_pred_binary = (prob_target >= t).astype(int)
#         tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
#
#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#
#         sens.append(sensitivity)
#         spec.append(specificity)
#
#     balanced_score = np.array(sens) + np.array(spec)
#     best_idx = np.argmax(balanced_score)
#
#     plt.scatter(sens[best_idx], spec[best_idx], label=target_class)
#
# plt.xlabel("Sensitivity")
# plt.ylabel("Specificity")
# plt.title("Random Forest - Optimal Sensitivity vs Specificity per Class")
# plt.legend()
# plt.grid(True)
# plt.show()
# # ============================================================
# # Optimal Threshold per Class (Youden's Index)
# # Compare Default 0.5 vs Optimized Threshold
# # ============================================================
#
# import numpy as np
# import pandas as pd
# from sklearn.metrics import confusion_matrix
#
# def threshold_analysis(model, X_test, y_test, model_name):
#
#     print(f"\n{'='*60}")
#     print(f"Threshold Optimization Results - {model_name}")
#     print(f"{'='*60}\n")
#
#     classes = model.classes_
#     prob = model.predict_proba(X_test)
#
#     thresholds = np.linspace(0, 1, 101)
#
#     results = []
#
#     for i, cls in enumerate(classes):
#
#         y_true_binary = (y_test == cls).astype(int)
#         prob_target = prob[:, i]
#
#         best_youden = -1
#         best_threshold = 0.5
#         best_sens = 0
#         best_spec = 0
#
#         # ----- Search best threshold -----
#         for t in thresholds:
#
#             y_pred_binary = (prob_target >= t).astype(int)
#
#             tn, fp, fn, tp = confusion_matrix(
#                 y_true_binary,
#                 y_pred_binary
#             ).ravel()
#
#             sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#             specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#
#             youden = sensitivity + specificity - 1
#
#             if youden > best_youden:
#                 best_youden = youden
#                 best_threshold = t
#                 best_sens = sensitivity
#                 best_spec = specificity
#
#         # ----- Default threshold (0.5) -----
#         y_pred_default = (prob_target >= 0.5).astype(int)
#         tn, fp, fn, tp = confusion_matrix(
#             y_true_binary,
#             y_pred_default
#         ).ravel()
#
#         sens_default = tp / (tp + fn) if (tp + fn) > 0 else 0
#         spec_default = tn / (tn + fp) if (tn + fp) > 0 else 0
#
#         results.append({
#             "Class": cls,
#
#             "Default Threshold": 0.5,
#             "Default Sensitivity": round(sens_default, 3),
#             "Default Specificity": round(spec_default, 3),
#
#             "Best Threshold": round(best_threshold, 3),
#             "Optimized Sensitivity": round(best_sens, 3),
#             "Optimized Specificity": round(best_spec, 3),
#
#             "Youden Index": round(best_youden, 3)
#         })
#
#     df_results = pd.DataFrame(results)
#     print(df_results)
#
#     return df_results
#
#
# # ==========================
# # Run for Logistic Regression
# # ==========================
# threshold_results_log = threshold_analysis(
#     best_log,
#     X_test,
#     y_test,
#     "Logistic Regression"
# )
#
# # ==========================
# # Run for Random Forest
# # ==========================
# threshold_results_rf = threshold_analysis(
#     best_rf,
#     X_test,
#     y_test,
#     "Random Forest"
# )
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
# # Example results from your threshold optimization
# classes = ["Cloud Engineer", "Data Analyst", "Data Scientist", "Machine Learning Engineer", "Software Engineer"]
#
# youden_lr = [0.901, 0.981, 0.946, 0.905, 0.920]
# youden_rf = [0.953, 0.979, 0.972, 0.917, 0.935]
#
# df = pd.DataFrame({
#     "Class": classes,
#     "Logistic Regression": youden_lr,
#     "Random Forest": youden_rf
# })
#
# # Plotting
# x = np.arange(len(classes))  # the label locations
# width = 0.35  # width of the bars
#
# fig, ax = plt.subplots(figsize=(10,6))
# rects1 = ax.bar(x - width/2, df["Logistic Regression"], width, label='Logistic Regression', color="#1f77b4")
# rects2 = ax.bar(x + width/2, df["Random Forest"], width, label='Random Forest', color="#ff7f0e")
#
# # Add labels and title
# ax.set_ylabel("Youden Index")
# ax.set_xlabel("Job Role")
# ax.set_title("Youden Index per Class: Logistic Regression vs Random Forest")
# ax.set_xticks(x)
# ax.set_xticklabels(df["Class"], rotation=30)
# ax.set_ylim(0, 1.05)
# ax.legend()
# ax.grid(axis='y', linestyle='--', alpha=0.7)
#
# # Annotate bars with values
# for rects in [rects1, rects2]:
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height:.3f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0,3),
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=9)
#
# plt.tight_layout()
# plt.show()
