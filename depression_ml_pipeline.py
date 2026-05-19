# ==========================================
# Student Depression ML Pipeline
# ==========================================

# 1. Install Dependencies & Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# 2. Data Loading and Cleaning (Section 3.2 of Technical Report)
file_name = 'student_depression_dataset.csv'
print(f"Loading '{file_name}' from storage...")

try:
    df = pd.read_csv(file_name)
    print(f"\n[SUCCESS] Dataset successfully loaded!")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Data Cleaning & Profiling
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        print("-> Cleaned: Removed 'id' column.")

    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        print(f"-> Detected {null_counts} missing value(s) in the dataset.")
        df = df.dropna()
        print(f"-> Cleaned: Dropped rows with missing values. New Shape: {df.shape}")
    else:
        print("-> Cleaned: Checked for missing values. None detected.")

    print("\n--- First 5 Records of Cleaned Dataset ---")
    print(df.head())

except FileNotFoundError:
    print(f"\n[ERROR] The file '{file_name}' was not found. Please upload it to the directory.")
    # Exit script if dataset cannot be loaded
    import sys
    sys.exit()

# 3. Exploratory Data Analysis (EDA)
# Set clean, professional visual themes
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

# Chart 1: Academic Pressure vs. Depression Risk
print("\nGenerating Plot 1: Academic Pressure vs. Depression...")
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    x='Academic Pressure',
    hue='Depression',
    palette='viridis'
)
plt.title('Depression Status Grouped by Academic Pressure (Scale 1-5)', pad=15, fontweight='bold')
plt.xlabel('Academic Pressure Level')
plt.ylabel('Student Count')
plt.legend(title='Depression Status', labels=['No Risk', 'At Risk'])
plt.tight_layout()
plt.savefig('academic_pressure_vs_depression.png', dpi=300)
plt.show()

# Chart 2: Sleep Duration vs. Depression Risk
print("\nGenerating Plot 2: Sleep Duration vs. Depression...")
plt.figure(figsize=(10, 6))

# Check unique categories to avoid sorting errors
sleep_order = df['Sleep Duration'].unique().tolist()
standard_order = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
order_to_use = [o for o in standard_order if o in sleep_order] or sleep_order

sns.countplot(
    data=df,
    x='Sleep Duration',
    hue='Depression',
    palette='mako',
    order=order_to_use
)
plt.title('Depression Rates Sorted by Sleep Duration', pad=15, fontweight='bold')
plt.xlabel('Sleep Duration')
plt.ylabel('Student Count')
plt.legend(title='Depression Status', labels=['No Risk', 'At Risk'])
plt.tight_layout()
plt.savefig('sleep_duration_vs_depression.png', dpi=300)
plt.show()

print("\n[SUCCESS] Visualizations successfully plotted and saved to disk as PNG files!")

# 4. Preprocessing & Feature Engineering (Section 3.2 of Technical Report)
print("\n--- Starting Preprocessing & Feature Engineering Phase ---")

# Create a copy of the dataframe for transformation
processed_df = df.copy()

# Identify and separate feature groups
categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = processed_df.select_dtypes(exclude=['object']).columns.tolist()

# Remove our target 'Depression' from the list of numerical features to scale
if 'Depression' in numerical_cols:
    numerical_cols.remove('Depression')
elif 'Depression' in categorical_cols:
    categorical_cols.remove('Depression')

print(f"Categorical features detected: {categorical_cols}")
print(f"Numerical features detected: {numerical_cols}")

# Apply Label Encoding to categorical fields
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Strip any extra quotes or spacing
    processed_df[col] = processed_df[col].astype(str).str.replace('"', '').str.strip()
    processed_df[col] = le.fit_transform(processed_df[col])
    encoders[col] = le

# Split into features (X) and target (y)
X = processed_df.drop(columns=['Depression'])
y = processed_df['Depression']

# Scale continuous features using Standard Scaler
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"\n[SUCCESS] Split completed successfully!")
print(f"Training Set Shape: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Validation Set Shape: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# 5. Model Training & Hyperparameter Tuning (Section 4.3 of Technical Report)
print("\n--- Starting Model Training & Optimization ---")
print("Configuring XGBoost with 5-Fold Grid Search (Optimizing for Recall)...")

# Initialize base classifier
xgb_base = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200]
}

# Define cross-validation grid search
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1
)

# Fit models
grid_search.fit(X_train, y_train)

# Select the best estimator
best_xgb = grid_search.best_estimator_

print("\n[SUCCESS] XGBoost Model Training Complete!")
print(f"Optimal Hyperparameters: {grid_search.best_params_}")

# 6. Evaluation & Diagnostic Plots (Section 5.0 of Technical Report)
print("\n--- Starting Evaluation & Generating Diagnostic Plots ---")

# Generate predictions for evaluation plots
y_pred = best_xgb.predict(X_test)
y_prob = best_xgb.predict_proba(X_test)[:, 1]

# 1. Confusion Matrix (Section 5.3 of Technical Report)
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=['No Risk', 'At Risk'], 
    yticklabels=['No Risk', 'At Risk']
)
plt.title('Validation Set Confusion Matrix', pad=15, fontweight='bold')
plt.xlabel('Predicted Diagnosis Label')
plt.ylabel('Actual Clinical Assessment')
plt.tight_layout()
plt.savefig('confusion_matrix_evaluation.png', dpi=300)
plt.show()

# 2. Feature Importance Plot (Section 5.3 of Technical Report)
print("\nGenerating Feature Importance Plot...")
importances = best_xgb.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 7))
sns.barplot(
    x=importances[indices],
    y=feature_names[indices],
    hue=feature_names[indices],
    palette='viridis',
    legend=False
)
plt.title('XGBoost Multi-Factor Feature Importance Ranking', pad=15, fontweight='bold', fontsize=14)
plt.xlabel('Information Gain Weight', labelpad=10, fontsize=12)
plt.ylabel('Lifestyle, Social, & Academic Features', labelpad=10, fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance_analysis.png', dpi=300)
plt.show()

# 3. Receiver Operating Characteristic (ROC) Curve (Section 5.3 of Technical Report)
print("\nGenerating ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:0.3f})')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', pad=15, fontweight='bold', fontsize=14)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve_evaluation.png', dpi=300)
plt.show()

print("\n[SUCCESS] Evaluation completed. All metrics calculated and plots generated!")