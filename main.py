import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#! Load the data
data = pd.read_csv('data/data.csv')
#! Numerical columns
numerical_cols = data.select_dtypes(include=[float, int]).columns.tolist()
print(numerical_cols)
#! Categorical columns
categorical_cols = data.select_dtypes(include=[object]).columns.tolist()
print(categorical_cols)
#! Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(data['Temparature'], data['Crop Type'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#! Split the data
