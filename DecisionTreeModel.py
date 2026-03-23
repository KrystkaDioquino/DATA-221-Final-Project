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
