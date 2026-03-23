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

# DECISION TREE REGRESSOR MODEL

# Try different max_depth values to find best performance
depths = [2, 3, 5, 10, None]

print("\nDecision Tree Results:\n")

for d in depths:
    model = DecisionTreeRegressor(max_depth=d, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    train_r2 = r2_score(y_train, y_train_pred)

    print(f"Max Depth: {d}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"Test R2: {r2}")
    print(f"Train R2: {train_r2}")
    print("------------------------")