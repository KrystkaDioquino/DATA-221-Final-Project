import pandas as pd
import matplotlib.pyplot as plt

# Our data
evaluation_metrics = {
    'R2': [0.19, 0.14, 0.09, 0.19],
    'RMSE': [14.05, 14.44, 14.85, 14.07],
    'MAE': [11.16, 11.38, 11.81, 10.96]
}
model_names = ['Linear Regression', 'K Nearest Neighbour', 'Decision Tree', 'Neural Network']

# Makes the table
df = pd.DataFrame(evaluation_metrics, index=model_names)
# Creates the bar chart
df.plot(kind='bar', color=['skyblue', 'gold', 'pink'])


plt.xlabel('Models')
plt.ylabel('Performance Score / Error')
plt.legend(title='Metrics')
plt.title('Performance Metrics Comparison Against Each Model')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# load the dataset into a table
df = pd.read_csv("StudentsPerformance.csv")

# check for missing values and duplicates
print("Missing values:\n", df.isna().sum())
print("Duplicates:", df.duplicated().any())

# remove reading and writing scores to prevent data leakage
df = df.drop(columns=["reading score", "writing score"])

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

# separate inputs (X) from the target we want to predict (y)
X = df.drop("math score", axis=1)
y = df["math score"]

# split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features so they all have equal influence on the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create and train the multiple linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# use the trained model to make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# evaluate on training set
print("\ntraining set:")
print("mae =", round(mean_absolute_error(y_train, y_train_pred), 4))
print("rmse =", round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4))
print("r2 =", round(r2_score(y_train, y_train_pred), 4))

# evaluate on testing set
print("\ntesting set:")
print("mae =", round(mean_absolute_error(y_test, y_test_pred), 4))
print("rmse =", round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4))
print("r2 =", round(r2_score(y_test, y_test_pred), 4))

# if training and testing results are close, the model is not overfitting.