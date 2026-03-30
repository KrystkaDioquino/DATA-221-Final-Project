import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# DATASET PREPARATION:

# load the dataset into a table
df = pd.read_csv("StudentsPerformance.csv")

# check for missing values and duplicates
print("Missing values:\n", df.isna().sum())
print("Duplicates:", df.duplicated().any())


# DATA CLEANING:

# remove reading and writing scores to prevent data leakage
df = df.drop(columns=["reading score", "writing score"])


# FEATURE ENCODING:

# parental education has a natural order so we assign numbers to reflect that
education_hierarchy = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}
df["parental level of education"] = df["parental level of education"].map(education_hierarchy)

# convert remaining categorical columns to numbers using one-hot encoding
df = pd.get_dummies(df, columns=["gender", "race/ethnicity", "lunch", "test preparation course"], drop_first=True)


# DATA SPLITTING:

# separate inputs (X) from the target we want to predict (y)
X = df.drop("math score", axis=1)
y = df["math score"]

# split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# FEATURE SCALING:

# scale features so they all have equal influence on the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# MODEL A - WITHOUT READING AND WRITING SCORES:

# create and train the multiple linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# use the trained model to make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("\nMODEL A (No Reading/Writing Scores):")

print("\nTraining Set:")
print("MAE =", round(mean_absolute_error(y_train, y_train_pred), 4))
print("RMSE =", round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4))
print("R2 =", round(r2_score(y_train, y_train_pred), 4))

print("\nTesting Set:")
print("MAE =", round(mean_absolute_error(y_test, y_test_pred), 4))
print("RMSE =", round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4))
print("R2 =", round(r2_score(y_test, y_test_pred), 4))

# if training and testing results are close, the model is not overfitting


# MODEL B - WITH READING AND WRITING SCORES (COMPARISON):

# load the dataset again to keep reading and writing scores
df_b = pd.read_csv("StudentsPerformance.csv")

# apply the same encoding as model a
df_b["parental level of education"] = df_b["parental level of education"].map(education_hierarchy)
df_b = pd.get_dummies(df_b, columns=["gender", "race/ethnicity", "lunch", "test preparation course"], drop_first=True)

# separate inputs and target
X_b = df_b.drop("math score", axis=1)
y_b = df_b["math score"]

# use the same split to keep comparison fair
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, test_size=0.2, random_state=42)

# scale the features
scaler_b = StandardScaler()
X_b_train_scaled = scaler_b.fit_transform(X_b_train)
X_b_test_scaled = scaler_b.transform(X_b_test)

# train the comparison model
model_b = LinearRegression()
model_b.fit(X_b_train_scaled, y_b_train)

# make predictions
y_b_train_pred = model_b.predict(X_b_train_scaled)
y_b_test_pred = model_b.predict(X_b_test_scaled)

print("\nMODEL B (With Reading/Writing Scores):")

print("\nTraining Set:")
print("MAE =", round(mean_absolute_error(y_b_train, y_b_train_pred), 4))
print("RMSE =", round(np.sqrt(mean_squared_error(y_b_train, y_b_train_pred)), 4))
print("R2 =", round(r2_score(y_b_train, y_b_train_pred), 4))

print("\nTesting Set:")
print("MAE =", round(mean_absolute_error(y_b_test, y_b_test_pred), 4))
print("RMSE =", round(np.sqrt(mean_squared_error(y_b_test, y_b_test_pred)), 4))
print("R2 =", round(r2_score(y_b_test, y_b_test_pred), 4))