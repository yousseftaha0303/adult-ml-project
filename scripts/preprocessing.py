import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv(r'data\adult.csv', na_values='?')
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# --------------------------
# 1. Fill missing data
# --------------------------
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)

# --------------------------
# 2. Preprocessing for Clustering
# --------------------------
data_clustering = data.copy()
for col in ["workclass", "occupation", "native-country"]:
    data_clustering[col].fillna("Unknown", inplace=True)

# Encode categorical variables
categorical_cols = data_clustering.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in categorical_cols:
    data_clustering[col] = le.fit_transform(data_clustering[col])

# Standardize numeric columns
numeric_cols = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
scaler_cluster = StandardScaler()
data_clustering[numeric_cols] = scaler_cluster.fit_transform(data_clustering[numeric_cols])

# Optionally drop target for unsupervised clustering
if 'income' in data_clustering.columns:
    data_clustering_no_target = data_clustering.drop(columns=['income'])
else:
    data_clustering_no_target = data_clustering.copy()

# --------------------------
# 3. Preprocessing for Classification
# --------------------------
data_classification = pd.get_dummies(data, drop_first=True)

X_class = data_classification.drop("income_>50K", axis=1)
y_class = data_classification["income_>50K"]

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=10
)

scaler_class = StandardScaler()
X_train_class[numeric_cols] = scaler_class.fit_transform(X_train_class[numeric_cols])
X_test_class[numeric_cols] = scaler_class.transform(X_test_class[numeric_cols])

# --------------------------
# 4. Preprocessing for Regression
# --------------------------
# predict 'hours-per-week' (numeric target)
data_regression = pd.get_dummies(data.drop(columns=['hours-per-week']), drop_first=True)
y_reg = data['hours-per-week']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    data_regression, y_reg, test_size=0.2, random_state=10
)

scaler_reg = StandardScaler()
X_train_reg[numeric_cols[:-1]] = scaler_reg.fit_transform(X_train_reg[numeric_cols[:-1]])
X_test_reg[numeric_cols[:-1]] = scaler_reg.transform(X_test_reg[numeric_cols[:-1]])
