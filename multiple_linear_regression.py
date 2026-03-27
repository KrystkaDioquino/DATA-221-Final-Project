import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("StudentsPerformance.csv")

print("Missing values:\n", df.isna().sum())
print("Duplicates:", df.duplicated().any())

df = df.drop(columns=["reading score", "writing score"])

education_hierarchy = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}
df["parental level of education"] = df["parental level of education"].map(education_hierarchy)

df = pd.get_dummies(df, columns=["gender", "race/ethnicity", "lunch", "test preparation course"], drop_first=True)

X = df.drop("math score", axis=1)
y = df["math score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("TRAINING SET =")
print("MAE:", round(mean_absolute_error(y_train, y_train_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4))
print("R2:", round(r2_score(y_train, y_train_pred), 4))

print("\nTESTING SET =")
print("MAE:", round(mean_absolute_error(y_test, y_test_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4))
print("R2:", round(r2_score(y_test, y_test_pred), 4))