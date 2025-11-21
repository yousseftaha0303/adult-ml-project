import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ================================================================
# 1. DATA LOADING & INITIAL CLEANING
# ================================================================
print("="*60)
print("STEP 1: DATA LOADING & CLEANING")
print("="*60)

# Load data treating '?' as missing
data = pd.read_csv(r'data\adult.csv', na_values='?')

# Strip whitespace from ALL object columns FIRST
data = data.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

# NOW identify any remaining '?' that might have had spaces
data = data.replace('?', np.nan)

print(f"\n✓ Dataset loaded: {data.shape[0]} rows × {data.shape[1]} columns")
print(f"✓ Missing values found: {data.isnull().sum().sum()}")

print("\n--- DATA INFO ---")
print(data.info())

print("\n--- FIRST 5 ROWS ---")
print(data.head())

print("\n--- DESCRIPTIVE STATISTICS (NUMERICAL) ---")
print(data.describe())

print("\n--- CATEGORY VALUE COUNTS (income) ---")
print(data['income'].value_counts())

# ================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA) - BEFORE PREPROCESSING
# ================================================================
print("\n" + "="*60)
print("STEP 2: EXPLORATORY DATA ANALYSIS (RAW DATA)")
print("="*60)

# --- Missing Values Heatmap ---
plt.figure(figsize=(10,6))
sns.heatmap(data.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

# --- Income Distribution ---
plt.figure(figsize=(8,5))
data['income'].value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title("Income Distribution")
plt.xlabel("Income Class")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# --- Histograms of Numeric Features (BEFORE Transformation) ---
data.select_dtypes(include=['int64', 'float64']).hist(figsize=(12,10))
plt.suptitle("Histograms of Numerical Features (RAW DATA)")
plt.show()

# --- Count Plots of Categorical Variables ---
categorical = data.select_dtypes(include=['object']).columns
fig, axes = plt.subplots(2, 3, figsize=(16,10))
axes = axes.flatten()
for i, col in enumerate(categorical[:6]):
    sns.countplot(data=data, x=col, ax=axes[i])
    axes[i].set_title(f"Count Plot - {col}")
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# --- Scatter Plot: Age vs Hours-per-week ---
plt.figure(figsize=(10,6))
sns.scatterplot(data=data, x="age", y="hours.per.week", hue="income", alpha=0.6)
plt.title("Age vs Hours-per-week by Income")
plt.show()

# --- CORRELATION HEATMAP ---
plt.figure(figsize=(12, 8))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Numeric Features - RAW DATA)")
plt.show()

# --- BOX-PLOTS (OUTLIERS IDENTIFICATION) ---
numeric_cols = ["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]

plt.figure(figsize=(14, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(y=data[col])
    plt.title(f"Boxplot: {col}")
plt.tight_layout()
plt.show()

# --- Income Distribution by Selected Categorical Features ---
cat_features = ['education', 'marital.status', 'workclass', 'sex', 'race']
for col in cat_features:
    plt.figure(figsize=(10,5))
    sns.countplot(data=data, x=col, hue='income')
    plt.title(f"Income Distribution by {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Boxplots: Continuous vs Income ---
continuous_features = ['age', 'education.num', 'hours.per.week', 'capital.gain', 'capital.loss']
for col in continuous_features:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=data, x='income', y=col)
    plt.title(f"{col} Distribution by Income")
    plt.tight_layout()
    plt.show()

# --- Pairplot: Relationships Among Key Numeric Features ---
sns.pairplot(
    data,
    vars=["age","education.num","capital.gain","hours.per.week"],
    hue="income",
    plot_kws={'alpha':0.5, 's':30}
)
plt.suptitle("Pairwise Relationships – Colored by Income", y=1.02)
plt.show()

# --- Correlation Heatmaps by Income Group ---
plt.figure(figsize=(10,6))
sns.heatmap(
    data[data['income'] == '<=50K'].select_dtypes(include='number').corr(),
    cmap='coolwarm',
    center=0,
    annot=True
)
plt.title("Correlation Heatmap — <=50K Group")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(
    data[data['income'] == '>50K'].select_dtypes(include='number').corr(),
    cmap='coolwarm',
    center=0,
    annot=True
)
plt.title("Correlation Heatmap — >50K Group")
plt.show()

# --- Capital Gain / Loss Distribution ---
plt.figure(figsize=(8,4))
sns.histplot(data['capital.gain'], bins=60, color='green', kde=True)
plt.xlim(0, 20000)
plt.title("Capital Gain Distribution (Zoomed under 20,000)")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(data['capital.loss'], bins=60, color='red', kde=True)
plt.xlim(0, 2000)
plt.title("Capital Loss Distribution (Zoomed under 2,000)")
plt.show()

# ================================================================
# 3. HANDLE MISSING VALUES
# ================================================================
print("\n" + "="*60)
print("STEP 3: HANDLING MISSING VALUES")
print("="*60)

missing_cols = data.columns[data.isnull().any()].tolist()
print(f"\nColumns with missing values: {missing_cols}")

for col in missing_cols:
    mode_val = data[col].mode()[0]
    missing_count = data[col].isnull().sum()
    data[col].fillna(mode_val, inplace=True)
    print(f"  ✓ Imputed {missing_count} values in '{col}' with mode: '{mode_val}'")

print("\n✓ All missing values handled successfully")
print(f"✓ Remaining missing values: {data.isnull().sum().sum()}")

# ================================================================
# 4. FEATURE INSPECTION & REMOVAL
# ================================================================
print("\n" + "="*60)
print("STEP 4: FEATURE INSPECTION & REMOVAL")
print("="*60)

# Drop fnlwgt (survey weight)
if 'fnlwgt' in data.columns:
    data = data.drop('fnlwgt', axis=1)
    print("\n✓ Dropped 'fnlwgt' (survey weight)")

# Combine rare categories in workclass
print("\n--- Workclass value counts before combining ---")
print(data['workclass'].value_counts())

rare_workclass = ['Without-pay', 'Never-worked']
data['workclass'] = data['workclass'].replace(rare_workclass, 'Other')
print(f"\n✓ Combined rare workclass categories into 'Other'")
print(data['workclass'].value_counts())

# Group rare countries into 'Other'
print("\n--- Native country value counts (showing countries < 100) ---")
country_counts = data['native.country'].value_counts()
print(country_counts[country_counts < 100])

rare_countries = country_counts[country_counts < 100].index.tolist()
data['native.country'] = data['native.country'].replace(rare_countries, 'Other')
print(f"\n✓ Grouped {len(rare_countries)} rare countries into 'Other'")

# ================================================================
# 5. FEATURE TRANSFORMATION
# ================================================================
print("\n" + "="*60)
print("STEP 5: FEATURE TRANSFORMATION")
print("="*60)

print("\n--- Capital Gain/Loss before transformation ---")
print(f"Capital Gain - Mean: {data['capital.gain'].mean():.2f}, Max: {data['capital.gain'].max()}")
print(f"Capital Loss - Mean: {data['capital.loss'].mean():.2f}, Max: {data['capital.loss'].max()}")

# SAVE ORIGINAL VALUES BEFORE TRANSFORMATION
capital_gain_original = data['capital.gain'].copy()
capital_loss_original = data['capital.loss'].copy()

# Apply log transformation
data['capital.gain'] = np.log1p(data['capital.gain'])
data['capital.loss'] = np.log1p(data['capital.loss'])
print("\n✓ Applied log1p transformation to capital.gain and capital.loss")

print("\n--- Capital Gain/Loss after transformation ---")
print(f"Capital Gain - Mean: {data['capital.gain'].mean():.2f}, Max: {data['capital.gain'].max():.2f}")
print(f"Capital Loss - Mean: {data['capital.loss'].mean():.2f}, Max: {data['capital.loss'].max():.2f}")

# Create net capital feature
data['net_capital'] = data['capital.gain'] - data['capital.loss']
print("\n✓ Created 'net_capital' feature (capital.gain - capital.loss)")

# ================================================================
# VISUALIZATION: Before vs After Log Transformation
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Row 1: Capital Gain
# BEFORE (non-zero only to see the skew)
axes[0, 0].hist(capital_gain_original[capital_gain_original > 0], bins=50, 
                color='green', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Capital Gain - BEFORE log1p (Non-Zero Values)', fontweight='bold')
axes[0, 0].set_xlabel('Original Value ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_xlim(0, 25000)

# AFTER
axes[0, 1].hist(data['capital.gain'], bins=50, 
                color='green', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Capital Gain - AFTER log1p (All Values)', fontweight='bold')
axes[0, 1].set_xlabel('Log Transformed Value')
axes[0, 1].set_ylabel('Frequency')

# Row 2: Capital Loss
# BEFORE (non-zero only to see the skew)
axes[1, 0].hist(capital_loss_original[capital_loss_original > 0], bins=50, 
                color='red', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Capital Loss - BEFORE log1p (Non-Zero Values)', fontweight='bold')
axes[1, 0].set_xlabel('Original Value ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_xlim(0, 3000)

# AFTER
axes[1, 1].hist(data['capital.loss'], bins=50, 
                color='red', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Capital Loss - AFTER log1p (All Values)', fontweight='bold')
axes[1, 1].set_xlabel('Log Transformed Value')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
# ================================================================
# 6. PREPARE DATASETS FOR DIFFERENT TASKS
# ================================================================
print("\n" + "="*60)
print("STEP 6: PREPARING TASK-SPECIFIC DATASETS")
print("="*60)

# Define numeric columns (after transformations and drops)
numeric_cols = ['age', 'education.num', 'capital.gain', 'capital.loss', 
                'hours.per.week', 'net_capital']

# ----------------------------------------------------------------
# 6A. CLUSTERING DATASET (Label Encoding + Scaling)
# ----------------------------------------------------------------
print("\n--- Preprocessing for CLUSTERING ---")
data_clustering = data.copy()

# Label encode ALL columns
categorical_cols_cluster = data_clustering.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in categorical_cols_cluster:
    data_clustering[col] = le.fit_transform(data_clustering[col])

print(f"✓ Label encoded {len(categorical_cols_cluster)} categorical columns")

# Scale numeric features
scaler_cluster = StandardScaler()
data_clustering[numeric_cols] = scaler_cluster.fit_transform(data_clustering[numeric_cols])
print(f"✓ Scaled {len(numeric_cols)} numeric features")

# Remove income for unsupervised learning
data_clustering_no_target = data_clustering.drop(columns=['income'])

print(f"✓ Clustering data shape: {data_clustering_no_target.shape}")
print(f"✓ All features label-encoded and scaled")

# ----------------------------------------------------------------
# 6B. CLASSIFICATION DATASET (One-Hot Encoding + Scaling)
# ----------------------------------------------------------------
print("\n--- Preprocessing for CLASSIFICATION ---")

# One-hot encode the entire dataset
data_classification = pd.get_dummies(data, drop_first=True)
print(f"✓ One-hot encoded dataset shape: {data_classification.shape}")

# Separate features and target
X_class = data_classification.drop("income_>50K", axis=1)
y_class = data_classification["income_>50K"]

print(f"✓ Features shape: {X_class.shape}")
print(f"✓ Target distribution:\n{y_class.value_counts(normalize=True)}")

# Train-test split with stratification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print(f"✓ Train set: {X_train_class.shape}, Test set: {X_test_class.shape}")

# Scale numeric features (fit on train, transform both)
scaler_class = StandardScaler()
numeric_features_in_class = [col for col in numeric_cols if col in X_train_class.columns]

X_train_class_scaled = X_train_class.copy()
X_test_class_scaled = X_test_class.copy()

X_train_class_scaled[numeric_features_in_class] = scaler_class.fit_transform(
    X_train_class[numeric_features_in_class]
)
X_test_class_scaled[numeric_features_in_class] = scaler_class.transform(
    X_test_class[numeric_features_in_class]
)

print(f"✓ Scaled {len(numeric_features_in_class)} numeric features")
print(f"✓ Class balance in train: {y_train_class.value_counts(normalize=True).to_dict()}")

# Use scaled versions
X_train_class = X_train_class_scaled
X_test_class = X_test_class_scaled

# ----------------------------------------------------------------
# 6C. REGRESSION DATASET (Predicting hours.per.week)
# ----------------------------------------------------------------
print("\n--- Preprocessing for REGRESSION ---")

# Create copy and remove target and income
data_regression_prep = data.copy()
y_reg = data_regression_prep['hours.per.week']
data_regression_prep = data_regression_prep.drop(columns=['hours.per.week', 'income'])

# One-hot encode
data_regression = pd.get_dummies(data_regression_prep, drop_first=True)
print(f"✓ One-hot encoded regression features shape: {data_regression.shape}")

# Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    data_regression, y_reg, test_size=0.2, random_state=42
)

print(f"✓ Train set: {X_train_reg.shape}, Test set: {X_test_reg.shape}")

# Scale numeric features
scaler_reg = StandardScaler()
numeric_features_in_reg = [col for col in numeric_cols if col in X_train_reg.columns and col != 'hours.per.week']

X_train_reg_scaled = X_train_reg.copy()
X_test_reg_scaled = X_test_reg.copy()

X_train_reg_scaled[numeric_features_in_reg] = scaler_reg.fit_transform(
    X_train_reg[numeric_features_in_reg]
)
X_test_reg_scaled[numeric_features_in_reg] = scaler_reg.transform(
    X_test_reg[numeric_features_in_reg]
)

print(f"✓ Scaled {len(numeric_features_in_reg)} numeric features")

# Use scaled versions
X_train_reg = X_train_reg_scaled
X_test_reg = X_test_reg_scaled

# ================================================================
# 7. FINAL SUMMARY
# ================================================================
print("\n" + "="*60)
print("✅ PREPROCESSING PIPELINE COMPLETE")
print("="*60)
print(f"""
📊 DATASET SUMMARY:
   • Original shape: {data.shape}
   • Features engineered: log1p transformations, net_capital
   • Missing values: All imputed with mode
   • Rare categories: Combined into 'Other'
   • Dropped features: fnlwgt

🎯 READY-TO-USE DATASETS:
   1. CLUSTERING:  data_clustering_no_target ({data_clustering_no_target.shape})
   2. CLASSIFICATION: 
      - X_train_class: {X_train_class.shape}
      - X_test_class: {X_test_class.shape}
      - y_train_class: {y_train_class.shape}
      - y_test_class: {y_test_class.shape}
   3. REGRESSION: 
      - X_train_reg: {X_train_reg.shape}
      - X_test_reg: {X_test_reg.shape}
      - y_train_reg: {y_train_reg.shape}
      - y_test_reg: {y_test_reg.shape}

🚀 Next Steps:
   → Run clustering algorithms (K-Means, DBSCAN)
   → Train classifiers (Logistic Regression, Random Forest, XGBoost)
   → Build regression models (Linear Regression, Gradient Boosting)
""")

print("\n✓ Data preprocessing completed successfully.")