import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Use relative path to the CSV file in your PyCharm project folder
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

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_acc = []
test_acc = []
max_iter_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

for iters in max_iter_list:
    lr = LogisticRegression(max_iter=iters, solver='lbfgs')
    lr.fit(X_train, y_train)

    # Track training accuracy
    y_train_pred = lr.predict(X_train)
    train_acc.append(accuracy_score(y_train, y_train_pred))

    # Track testing accuracy
    y_test_pred = lr.predict(X_test)
    test_acc.append(accuracy_score(y_test, y_test_pred))

plt.figure(figsize=(8, 6))
plt.plot(max_iter_list, train_acc, label="Train Accuracy", marker='o')
plt.plot(max_iter_list, test_acc, label="Test Accuracy", marker='o')
plt.xlabel("Max Iterations")
plt.ylabel("Accuracy")
plt.title("Logistic Regression: Train vs Test Accuracy")
plt.legend()
plt.grid(True)
plt.show()

train_acc_rf = []
test_acc_rf = []

rf = RandomForestClassifier(n_estimators=10, warm_start=True, random_state=42, class_weight='balanced')

for n_trees in range(10, 310, 10):
    rf.n_estimators = n_trees
    rf.fit(X_train, y_train)

    # Track training accuracy
    y_train_pred_rf = rf.predict(X_train)
    train_acc_rf.append(accuracy_score(y_train, y_train_pred_rf))

    # Track testing accuracy
    y_test_pred_rf = rf.predict(X_test)
    test_acc_rf.append(accuracy_score(y_test, y_test_pred_rf))

plt.figure(figsize=(8, 6))
plt.plot(range(10, 310, 10), train_acc_rf, label="Train Accuracy", marker='o')
plt.plot(range(10, 310, 10), test_acc_rf, label="Test Accuracy", marker='o')
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("Random Forest: Train vs Test Accuracy")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import log_loss

train_loss = []
test_loss = []
max_iter_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

for iters in max_iter_list:
    lr = LogisticRegression(max_iter=iters, solver='lbfgs')
    lr.fit(X_train, y_train)

    # Predicted probabilities
    y_train_prob = lr.predict_proba(X_train)
    y_test_prob = lr.predict_proba(X_test)

    # Compute log loss
    train_loss.append(log_loss(y_train, y_train_prob))
    test_loss.append(log_loss(y_test, y_test_prob))

plt.figure(figsize=(8, 6))
plt.plot(max_iter_list, train_loss, label="Train Loss", marker='o')
plt.plot(max_iter_list, test_loss, label="Test Loss", marker='o')
plt.xlabel("Max Iterations")
plt.ylabel("Log Loss")
plt.title("Logistic Regression: Train vs Test Loss")
plt.legend()
plt.grid(True)
plt.show()

train_loss_rf = []
test_loss_rf = []

rf = RandomForestClassifier(n_estimators=10, warm_start=True, random_state=42, class_weight='balanced')

for n_trees in range(10, 310, 10):
    rf.n_estimators = n_trees
    rf.fit(X_train, y_train)

    # Use 1 - accuracy as "loss"
    train_loss_rf.append(1 - accuracy_score(y_train, rf.predict(X_train)))
    test_loss_rf.append(1 - accuracy_score(y_test, rf.predict(X_test)))

plt.figure(figsize=(8, 6))
plt.plot(range(10, 310, 10), train_loss_rf, label="Train Loss", marker='o')
plt.plot(range(10, 310, 10), test_loss_rf, label="Test Loss", marker='o')
plt.xlabel("Number of Trees")
plt.ylabel("Error (1 - Accuracy)")
plt.title("Random Forest: Train vs Test Error")
plt.legend()
plt.grid(True)
plt.show()

