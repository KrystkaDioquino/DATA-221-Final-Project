import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



# ---------- DATASET PREPARATION ----------
# Load the dataset into pandas DataFrame
df = pd.read_csv("StudentsPerformance.csv")

# Dataframe with reading and writing score for model comparison
df_with_rw = df.copy()



# ---------- DATA CLEANING AND PREPROCESSING -------------
# Check for missing values in each column
# Ensures data completeness before processing
print("Missing values:\n", df.isna().sum())

# Check for duplicate rows in the dataset
# Helps avoid biased or repeated data
print("Duplicates:", df.duplicated().any())

# Remove reading and writing scores to avoid data leakage
df = df.drop(columns=["reading score", "writing score"])



# ---------- FEATURE ENCODING ----------
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
df_with_rw["parental level of education"] = df_with_rw["parental level of education"].map(education_hierarchy)

# Apply one-hot encoding to categorical variables without order
df = pd.get_dummies(df, columns=["gender", "race/ethnicity", "lunch", "test preparation course"], drop_first=True, dtype=int)
df_with_rw = pd.get_dummies(df_with_rw, columns=["gender", "race/ethnicity", "lunch", "test preparation course"],
                            drop_first=True, dtype=int)



# ---------- CORRELATION ANALYSIS ----------
plt.figure(figsize=(10,6))
ax = sns.heatmap(df_with_rw.corr(), annot=True, fmt=".2f",annot_kws={"size":7}, cmap="Reds")
plt.title("Correlation Matrix", fontsize=18)
plt.tight_layout()
plt.show()


# ---------- DATA SPLITTING ----------
# Separate feature matrix (X) and target variable (y)
X = df.drop("math score", axis=1)
y = df["math score"]

X_rw = df_with_rw.drop("math score", axis=1)
y_rw = df_with_rw["math score"]

# Split the dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_rw, X_test_rw, y_train_rw, y_test_rw = train_test_split(
    X_rw, y_rw, test_size=0.2, random_state=42)



# ---------- FEATURE SCALING ----------
# Initialize StandardScaler to normalize feature values
scaler = StandardScaler()

# Fit scaler on training data, then apply to test data
# Use on Multiple Linear Regression, KNN Regressor and Neural Network Regressor only
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_rw_scaled = scaler.fit_transform(X_train_rw)
X_test_rw_scaled = scaler.transform(X_test_rw)


# ------------ MODEL TRAINING ------------
# Decision Tree
model_A = DecisionTreeRegressor(max_depth=3, random_state=42)
model_A.fit(X_train, y_train)
pred_A = model_A.predict(X_test)


# Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_test_pred = model.predict(X_test_scaled)

# KNN
mae_list = []
for k in range(1, 101):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    mae_list.append(mean_absolute_error(y_test, knn.predict(X_test_scaled)))

best_k = np.argmin(mae_list) + 1
knn_model = KNeighborsRegressor(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

#Neural Network
nn_model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dense(1)])

nn_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
history = nn_model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=32)

# ---------- MODEL PREDICTIONS ----------
# Decision Tree predictions
y_pred_dt = model_A.predict(X_test)

# Linear Regression predictions
y_pred_lr = model.predict(X_test_scaled)

# KNN predictions
y_pred_knn = knn_model.predict(X_test_scaled)

# Neural Network predictions
y_pred_nn = nn_model.predict(X_test_scaled).flatten()


# ---------- PREDICTED VS ACTUAL PLOTS ----------
# Linear Regression
plt.figure()
plt.scatter(y_test, y_pred_lr, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression: Predicted vs Actual")
plt.show()

# KNN
plt.figure()
plt.scatter(y_test, y_pred_knn, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("KNN: Predicted vs Actual")
plt.show()

# Decision Tree
plt.figure()
plt.scatter(y_test, y_pred_dt, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Decision Tree: Predicted vs Actual")
plt.show()

# Neural Network
plt.figure()
plt.scatter(y_test, y_pred_nn, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Neural Network: Predicted vs Actual")
plt.show()


# ---------- RESIDUAL PLOTS ----------
# Linear Regression
plt.figure()
plt.scatter(y_pred_lr, y_test - y_pred_lr, alpha=0.6)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Linear Regression: Residual Plot")
plt.show()

# KNN
plt.figure()
plt.scatter(y_pred_knn, y_test - y_pred_knn, alpha=0.6)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("KNN: Residual Plot")
plt.show()

# Decision Tree
plt.figure()
plt.scatter(y_pred_dt, y_test - y_pred_dt, alpha=0.6)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Decision Tree: Residual Plot")
plt.show()

# Neural Network
plt.figure()
plt.scatter(y_pred_nn, y_test - y_pred_nn, alpha=0.6)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Neural Network: Residual Plot")
plt.show()
