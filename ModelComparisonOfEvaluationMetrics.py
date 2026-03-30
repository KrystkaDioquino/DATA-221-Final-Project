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
