import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('adult.csv', na_values='?') # reading the data and changing the '?' symobol to nan
data.head()
data.info()
data.describe()
data.isnull().sum()

#   Filling missing data

#   Option 1 (filling with mode) as it is a categorical data

for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)

#   Option 2 (Drop all the rows that has null values) incorrect as there are 2000 rows at least will be dropped

#data = data.dropna()      # not recommended 

#####     Some values contain leading spaces

data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

######## For Clustering we can make the NaN values be at class Unknown

data_clustering = data.copy()
for col in ["workclass", "occupation", "native-country"]:
    data_clustering[col].fillna("Unknown", inplace=True)

####      Encoding Categorical Variables for regeression and classification

data_classification = pd.get_dummies(data, drop_first=True)

########     Split X and Y         (Feature and Output)

X = data_classification.drop("income_>50K", axis=1)
y = data_classification["income_>50K"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

##           Standardization     (Z-score)
numeric_cols = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

####      Encoding Categorical Variables for Clustring

categorical_cols = data_clustering.select_dtypes(include=["object"]).columns
le = LabelEncoder()

for col in categorical_cols:
    data_clustering[col] = le.fit_transform(data_clustering[col])

##           Standardization     (Z-score)
data_clustering[numeric_cols] = scaler.fit_transform(data_clustering[numeric_cols])
