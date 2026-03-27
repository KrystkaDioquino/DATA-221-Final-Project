import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset as DataFrame
df = pd.read_csv("StudentsPerformance.csv")

# Data cleaning and processing
print("Missing values:\n", df.isna().sum())
print("Duplicates:", df.duplicated().any())

# ENCODING (Same for both versions)
education_hierarchy = {
    "some high school": 0, "high school": 1, "some college": 2,
    "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
}
df["parental level of education"] = df["parental level of education"].map(education_hierarchy)

# One-hot encoding for the rest
categorical_cols = ["gender", "race/ethnicity", "lunch", "test preparation course"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ==========================================
# VERSION A: WITHOUT Reading/Writing (Main)
# ==========================================
X_A = df_encoded.drop(columns=["math score", "reading score", "writing score"])
y_A = df_encoded["math score"]

X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)

# KNN needs scaling
scaler_A = StandardScaler()
X_train_A_scaled = scaler_A.fit_transform(X_train_A)
X_test_A_scaled = scaler_A.transform(X_test_A)

# Find Best K for Version A
mae_A_list = []
for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_A_scaled, y_train_A)
    mae_A_list.append(mean_absolute_error(y_test_A, knn.predict(X_test_A_scaled)))

best_k_A = np.argmin(mae_A_list) + 1
final_model_A = KNeighborsRegressor(n_neighbors=best_k_A)
final_model_A.fit(X_train_A_scaled, y_train_A)
pred_A = final_model_A.predict(X_test_A_scaled)

# ==========================================
# VERSION B: WITH Reading/Writing (Comparison)
# ==========================================
X_B = df_encoded.drop(columns=["math score"])
y_B = df_encoded["math score"]

X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=42)

scaler_B = StandardScaler()
X_train_B_scaled = scaler_B.fit_transform(X_train_B)
X_test_B_scaled = scaler_B.transform(X_test_B)

# Find Best K for Version B
mae_B_list = []
for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_B_scaled, y_train_B)
    mae_B_list.append(mean_absolute_error(y_test_B, knn.predict(X_test_B_scaled)))

best_k_B = np.argmin(mae_B_list) + 1
final_model_B = KNeighborsRegressor(n_neighbors=best_k_B)
final_model_B.fit(X_train_B_scaled, y_train_B)
pred_B = final_model_B.predict(X_test_B_scaled)

# ==========================================
# FINAL COMPARISON OUTPUT
# ==========================================
print("\n===== COMPARISON RESULTS =====")
print(f"\nVERSION A (No Reading/Writing) - Best K: {best_k_A}")
print(f"MAE: {mean_absolute_error(y_test_A, pred_A):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_A, pred_A)):.2f}")
print(f"R2: {r2_score(y_test_A, pred_A):.2f}")

print(f"\nVERSION B (With Reading/Writing) - Best K: {best_k_B}")
print(f"MAE: {mean_absolute_error(y_test_B, pred_B):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_B, pred_B)):.2f}")
print(f"R2: {r2_score(y_test_B, pred_B):.2f}")