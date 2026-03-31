## Student Math Performance Prediction

This project predicts student math exam scores using socioeconomic factors.
It explores how variables such as parental education, lunch type, and test preparation influence student performance.
___
# Project Overview
- Performed data cleaning & preprocessing (handling missing values, encoding categorical variables, preventing data leakage)
- Built and evaluated multiple models:
   1. Linear Regression
   2. K-Nearest Neighbors (KNN)
   3. Decision Tree
   4. Neural Network
- Compared performance using MAE, RMSE, and R² score
- Analyzed the impact of including vs excluding reading & writing scores

# Libraries
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

# Dataset

Dataset: Students Performance in Exams (Kaggle)

Features include:
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score
- Math Score (Target Variable)

# Model Evaluation

Metrics used:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)

---
# How to Run the Code
1. Clone the repository:
git clone https://github.com/KrystkaDioquino/DATA-221-Final-Project.git
cd DATA-221-Final-Project

2. Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

3. Run models separately:
- Linear Regression
python linear_regression_model.py

- KNN Model
python knn_model.py

- Decision Tree
python decision_tree_model.py

- Neural Network
python neural_network_model.py

# How to Reproduce Results
- Ensure the dataset file is in the project folder
- Run the script as provided (no changes needed)

The code will:
- Preprocess the data
- Train all models
- Output evaluation metrics (MAE, RMSE, R²)
- Generate visualizations (e.g., correlation heatmaps, predictions)
