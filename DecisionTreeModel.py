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


# ===============================
# VERSION A: WITHOUT reading/writing (MAIN MODEL)
# ===============================

df_A = df.drop(columns=["reading score", "writing score"])

# ENCODING CATEGORICAL VARIABLES
education_hierarchy = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}

df_A["parental level of education"] = df_A["parental level of education"].map(education_hierarchy)

df_A = pd.get_dummies(
    df_A,
    columns=["gender", "race/ethnicity", "lunch", "test preparation course"],
    drop_first=True
)

# DATA SPLITTING
X_A = df_A.drop("math score", axis=1)
y_A = df_A["math score"]

X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
    X_A, y_A, test_size=0.2, random_state=42
)

# TRAIN BEST MODEL (max_depth = 3 from earlier)
model_A = DecisionTreeRegressor(max_depth=3, random_state=42)
model_A.fit(X_train_A, y_train_A)

pred_A = model_A.predict(X_test_A)

# Evaluate A
mae_A = mean_absolute_error(y_test_A, pred_A)
rmse_A = np.sqrt(mean_squared_error(y_test_A, pred_A))
r2_A = r2_score(y_test_A, pred_A)


# ===============================
# VERSION B: WITH reading/writing (COMPARISON)
# ===============================

df_B = df.copy()

df_B["parental level of education"] = df_B["parental level of education"].map(education_hierarchy)

df_B = pd.get_dummies(
    df_B,
    columns=["gender", "race/ethnicity", "lunch", "test preparation course"],
    drop_first=True
)

# DATA SPLITTING
X_B = df_B.drop("math score", axis=1)
y_B = df_B["math score"]

X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X_B, y_B, test_size=0.2, random_state=42
)

# SAME MODEL
model_B = DecisionTreeRegressor(max_depth=3, random_state=42)
model_B.fit(X_train_B, y_train_B)

pred_B = model_B.predict(X_test_B)

# Evaluate B
mae_B = mean_absolute_error(y_test_B, pred_B)
rmse_B = np.sqrt(mean_squared_error(y_test_B, pred_B))
r2_B = r2_score(y_test_B, pred_B)


# ===============================
# FINAL COMPARISON OUTPUT
# ===============================

print("\n===== COMPARISON RESULTS =====\n")

print("WITHOUT reading/writing:")
print(f"MAE: {mae_A}")
print(f"RMSE: {rmse_A}")
print(f"R2: {r2_A}")

print("\nWITH reading/writing:")
print(f"MAE: {mae_B}")
print(f"RMSE: {rmse_B}")
print(f"R2: {r2_B}")