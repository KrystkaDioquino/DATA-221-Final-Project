import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DATASET PREPARATION

# Load the dataset into pandas DataFrame
df = pd.read_csv("StudentsPerformance.csv")


# DATA CLEANING AND PROCESSING

# Check for missing values in each column
# Ensures data completeness before processing
print("Missing values:\n",df.isna().sum())

# Check for duplicate rows in the dataset
# Helps avoid biased or repeated data
print("Duplicates: ",df.duplicated().any())

# Remove reading and writing scores to prevent data leakage
# These features are highly correlated with the target
df = df.drop(columns=["reading score", "writing score"])

# Encoding categorical variables to numerical

# Apply ordinal encoding to parental level of education to preserve the natural hierarchy of education levels
education_hierarchy = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5}

df["parental level of education"] = df["parental level of education"].map(education_hierarchy)

# Apply one-hot encoding to categorical variables without order
df = pd.get_dummies(df, columns=["gender", "race/ethnicity", "lunch", "test preparation course"], drop_first=True)


# DATA SPLITTING AND SCALING

# Separate feature matrix (X) and target variable (y)
X = df.drop("math score", axis=1)
y = df["math score"]

# Split the dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize StandardScaler to normalize feature values
scaler = StandardScaler()

# Fit scaler on training data, then apply to test data
# Use on Multiple Linear Regression, KNN Regressor and Neural Network Regressor only
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)