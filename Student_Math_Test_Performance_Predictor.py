import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TODO clean comments
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

# Dataframe with reading and writing score for model comparison
df_with_rw = df.copy()

# Remove reading and writing scores to prevent data leakage
# These features are highly correlated with the target
df = df.drop(columns=["reading score", "writing score"])

# =========================
# MODEL 1: WITHOUT reading/writing
# =========================
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
df = pd.get_dummies(df, columns=["gender", "race/ethnicity", "lunch", "test preparation course"], drop_first=True, dtype=int)

# DATA SPLITTING

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

neural_network_model = MLPRegressor(
    hidden_layer_sizes=(16,),
    activation="relu",
    solver="adam",
    alpha=5.0,
    learning_rate_init=0.005,
    max_iter=500,
    random_state=42)

neural_network_model.fit(X_train_scaled, y_train)

# Training predictions
y_train_pred = neural_network_model.predict(X_train_scaled)

# Testing predictions
y_test_pred = neural_network_model.predict(X_test_scaled)

# Training set evaluation
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

# Testing set evaluation
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

# Display training set evaluation
print("Neural Network Model (without reading/writing score)")
print("Training Set Evaluation")
print(f"MAE: {mae_train:.2f}")
print(f"RMSE: {rmse_train:.2f}")
print(f"R2: {r2_train:.2f}")

# Display testing set evaluation
print("\nTesting Set Evaluation")
print(f"MAE: {mae_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"R2: {r2_test:.2f}")

# =========================
# MODEL 2: WITH reading/writing
# =========================

df_with_rw["parental level of education"] = df_with_rw["parental level of education"].map(education_hierarchy)

# Apply one-hot encoding to categorical variables without order
df_with_rw = pd.get_dummies(df_with_rw, columns=["gender", "race/ethnicity", "lunch", "test preparation course"], drop_first=True, dtype=int)

# DATA SPLITTING

# Separate feature matrix (X) and target variable (y)
X_with_rw = df_with_rw.drop("math score", axis=1)
y_with_rw = df_with_rw["math score"]

X_train_with_rw, X_test_with_rw, y_train_with_rw, y_test_with_rw = train_test_split(X_with_rw, y_with_rw, test_size=0.2, random_state=42)

# Fit scaler on training data, then apply to test data
# Use on Multiple Linear Regression, KNN Regressor and Neural Network Regressor only
X_train_scaled_with_rw = scaler.fit_transform(X_train_with_rw)
X_test_scaled_with_rw = scaler.transform(X_test_with_rw)

neural_network_model_with_rw = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42)

# Train using the new _with_rw variables
neural_network_model_with_rw.fit(X_train_scaled_with_rw, y_train_with_rw)

# Training predictions
y_train_pred_with_rw = neural_network_model_with_rw.predict(X_train_scaled_with_rw)

# Testing predictions
y_test_pred_with_rw = neural_network_model_with_rw.predict(X_test_scaled_with_rw)

# Training set evaluation
mae_train_with_rw = mean_absolute_error(y_train_with_rw, y_train_pred_with_rw)
rmse_train_with_rw = np.sqrt(mean_squared_error(y_train_with_rw, y_train_pred_with_rw))
r2_train_with_rw = r2_score(y_train_with_rw, y_train_pred_with_rw)

# Testing set evaluation
mae_test_with_rw = mean_absolute_error(y_test_with_rw, y_test_pred_with_rw)
rmse_test_with_rw = np.sqrt(mean_squared_error(y_test_with_rw, y_test_pred_with_rw))
r2_test_with_rw = r2_score(y_test_with_rw, y_test_pred_with_rw)

# Display training set evaluation
print("\nNeural Network Model (with reading/writing score)")
print("Training Set Evaluation")
print(f"MAE: {mae_train_with_rw:.2f}")
print(f"RMSE: {rmse_train_with_rw:.2f}")
print(f"R2: {r2_train_with_rw:.2f}")

# Display testing set evaluation
print("\nTesting Set Evaluation")
print(f"MAE: {mae_test_with_rw:.2f}")
print(f"RMSE: {rmse_test_with_rw:.2f}")
print(f"R2: {r2_test_with_rw:.2f}")