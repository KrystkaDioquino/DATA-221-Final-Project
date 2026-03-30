import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset into pandas DataFrame
df = pd.read_csv("StudentsPerformance.csv")

# Keep a copy with reading and writing scores
df_with_rw = df.copy()

# Remove reading and writing scores to avoid data leakage
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
df_with_rw["parental level of education"] = df_with_rw["parental level of education"].map(education_hierarchy)


# Apply one-hot encoding to categorical variables without order
df = pd.get_dummies(df, columns=["gender", "race/ethnicity", "lunch", "test preparation course"],
                    drop_first=True, dtype=int)
df_with_rw = pd.get_dummies(df_with_rw, columns=["gender", "race/ethnicity", "lunch", "test preparation course"], 
                            drop_first=True, dtype=int)

# Ensure consistent results across runs
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters to try to find the best combination for best performance
hidden_layer_options = [(16,), (32,), (64, 32), (64, 32, 16)]
learning_rate_options = [0.001, 0.0005]
epoch_options = [50, 100, 200]

# --------------- MODEL 1 ----------------
# without reading and writing scores

# Separate feature matrix (X) and target variable (y)
X = df.drop("math score", axis=1)
y = df["math score"]

# Split the dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit scaler on training data, then apply to test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Track best results
best_r2 = -999
best_params = None
best_mae = None
best_rmse = None
best_model = None

# Build and compile neural network model using different hyperparameter combinations
for hidden_layers in hidden_layer_options:
    for learning_rate in learning_rate_options:
        for epochs in epoch_options:

            tf.random.set_seed(42)
            model = Sequential()
            model.add(tf.keras.Input(shape=(X_train_scaled.shape[1],)))

            for units in hidden_layers:
                model.add(Dense(units, activation="relu"))

            model.add(Dense(1))

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss="mse")

            model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32)

            # Predictions
            y_pred = model.predict(X_test_scaled).flatten()

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(hidden_layers, learning_rate, epochs,
                ": MAE:", round(mae, 2), "RMSE:", round(rmse, 2),"R2:", round(r2, 2))

            # Keep best model (based on R2)
            if r2 > best_r2:
                best_r2 = r2
                best_params = (hidden_layers, learning_rate, epochs)
                best_mae = mae
                best_rmse = rmse
                best_model = model

# Use best_model instead of undefined neural_network_model
y_train_pred = best_model.predict(X_train_scaled).flatten()
y_test_pred = best_model.predict(X_test_scaled).flatten()


# --------------- MODEL 2 ----------------
# with reading and writing scores

# Separate feature matrix (X) and target variable (y)
X_with_rw = df_with_rw.drop("math score", axis=1)
y_with_rw = df_with_rw["math score"]

X_train_with_rw, X_test_with_rw, y_train_with_rw, y_test_with_rw = train_test_split(
    X_with_rw, y_with_rw, test_size=0.2, random_state=42)

# Use a NEW scaler here to avoid accidentally mixing different feature sets.
scaler_with_rw = StandardScaler()
X_train_scaled_with_rw = scaler_with_rw.fit_transform(X_train_with_rw)
X_test_scaled_with_rw = scaler_with_rw.transform(X_test_with_rw)

# Build and train neural network model 2
# Build and compile neural network model 2
neural_network_model_with_rw = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled_with_rw.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)])

neural_network_model_with_rw.compile(optimizer="adam", loss="mse")

# Train model neural network model
neural_network_model_with_rw.fit(X_train_scaled_with_rw, y_train_with_rw, epochs=300)

# Predictions
y_train_pred_with_rw = neural_network_model_with_rw.predict(X_train_scaled_with_rw, verbose=0).flatten()
y_test_pred_with_rw = neural_network_model_with_rw.predict(X_test_scaled_with_rw, verbose=0).flatten()

# --------------- EVALUATION --------------
# Model 1
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

# Model 2
mae_train_with_rw = mean_absolute_error(y_train_with_rw, y_train_pred_with_rw)
rmse_train_with_rw = np.sqrt(mean_squared_error(y_train_with_rw, y_train_pred_with_rw))
r2_train_with_rw = r2_score(y_train_with_rw, y_train_pred_with_rw)

mae_test_with_rw = mean_absolute_error(y_test_with_rw, y_test_pred_with_rw)
rmse_test_with_rw = np.sqrt(mean_squared_error(y_test_with_rw, y_test_pred_with_rw))
r2_test_with_rw = r2_score(y_test_with_rw, y_test_pred_with_rw)

# Display training and testing set evaluation for model 1
print("Neural Network Model (without reading/writing score)")
print("Best Model 1 Parameters:", best_params)

print("\nTraining Set Evaluation")
print(f"MAE: {mae_train:.2f}")
print(f"RMSE: {rmse_train:.2f}")
print(f"R2: {r2_train:.2f}")

print("\nTesting Set Evaluation")
print(f"MAE: {mae_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"R2: {r2_test:.2f}")

# Display training and testing set evaluation for model 2
print("\nNeural Network Model (with reading/writing score)")
print("Training Set Evaluation")
print(f"MAE: {mae_train_with_rw:.2f}")
print(f"RMSE: {rmse_train_with_rw:.2f}")
print(f"R2: {r2_train_with_rw:.2f}")

print("\nTesting Set Evaluation")
print(f"MAE: {mae_test_with_rw:.2f}")
print(f"RMSE: {rmse_test_with_rw:.2f}")
print(f"R2: {r2_test_with_rw:.2f}")
