import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset into pandas DataFrame
df = pd.read_csv("StudentsPerformance.csv")

# Check for missing values in each column
# df.isna().sum() returns Series showing count of NaN per column
# StudentsPerformance.csv has 0 missing values
print("Missing values:\n",df.isna().sum())

# Check for any duplicated rows
# df.duplicated().any() returns True/False if any duplicate rows exist
print("Duplicates: ",df.duplicated().any())