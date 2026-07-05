import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Use the specific Pipeline tool from imblearn to prevent data leakage
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# 1. SIMULATING ENTERPRISE FINANCIAL DATASET
# (Replace this block with your actual dataset loading, e.g., pd.read_csv('creditcard.csv'))
np.random.seed(42)
n_samples = 284807
n_features = 29

X_dummy = np.random.randn(n_samples, n_features)
# Creating a highly imbalanced target array (99.83% Legitimate (0), 0.17% Fraudulent (1))
y_dummy = np.random.choice([0, 1], size=n_samples, p=[0.9983, 0.0017])

X = pd.DataFrame(X_dummy, columns=[f'V{i}' for i in range(1, n_features + 1)])
y = pd.Series(y_dummy)

# 2. LEAK-FREE TRAIN-TEST SPLIT
# We split first so the test set remains completely untouched by SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set class balance: \n{y_train.value_counts(normalize=True)}")
print(f"Testing set class balance: \n{y_test.value_counts(normalize=True)}\n")
print("-" * 60)

# 3. CONSTRUCTING THE LEAK-FREE PIPELINE (e.g., Random Forest Track)
# The imblearn pipeline automatically restricts SMOTE to the training calculations only.
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# 4. HYPERPARAMETER TUNING USING GRID SEARCH WITH CROSS-VALIDATION
# We define parameters to fine-tune our pipeline setup safely
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10]
}

print("Starting Hyperparameter Tuning & Pipeline Training...")
grid_search = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid,
    scoring='roc_auc',  # Strict evaluation metric target
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"\nBest Hyperparameters Found: {grid_search.best_params_}")
print("-" * 60)

# 5. STRICT EVALUATION ON UNTOUCHED ENTERPRISE TESTING DATA
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Generating Evaluation Metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

print("## FINAL PROJECT PERFORMANCE METRICS ##\n")
print(f"Strict ROC-AUC Score: {roc_auc:.4f} (Target: 0.85+)")
print("\nClassification Report (Focus on Precision & Recall):")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))

print("\nConfusion Matrix:")
print(f"True Negatives (Legitimate caught): {conf_matrix[0][0]}")
print(f"False Positives (False alarms):     {conf_matrix[0][1]}")
print(f"False Negatives (Missed Fraud!):    {conf_matrix[1][0]}")
print(f"True Positives (Fraud caught!):     {conf_matrix[1][1]}")