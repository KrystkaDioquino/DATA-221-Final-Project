import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# DATASET PREPARATION

# Load the dataset into pandas DataFrame
df = pd.read_csv("StudentsPerformance.csv")


# DATA CLEANING AND PROCESSING

# Check for missing values
print("Missing values:\n", df.isna().sum())

# Check for duplicate rows
print("Duplicates:", df.duplicated().any())

# Remove reading and writing scores to prevent data leakage
df = df.drop(columns=["reading score", "writing score"])

# ENCODING CATEGORICAL VARIABLES

# Ordinal encoding for parental education
education_hierarchy = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}

df["parental level of education"] = df["parental level of education"].map(education_hierarchy)

# One-hot encoding for remaining categorical variables
df = pd.get_dummies(
    df,
    columns=["gender", "race/ethnicity", "lunch", "test preparation course"],
    drop_first=True
)

# DATA SPLITTING

# Separate features (X) and target (y)
X = df.drop("math score", axis=1)
y = df["math score"]

# Split into training and testing sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)