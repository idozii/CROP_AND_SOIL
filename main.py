import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier

#! Load the data
data = pd.read_csv('data/data.csv')
#! Numerical columns
numerical_cols = data.select_dtypes(include=[float, int]).columns.tolist()
#! Categorical columns
categorical_cols = data.select_dtypes(include=[object]).columns.tolist()
#! Define thresholds for phosphorus levels
def phosphorous_level(p):
    if p > 30:
        return 'High'
    elif 15 <= p <= 30:
        return 'Medium'
    else:
        return 'Low'
data['Phosphorous'] = data['Phosphorous'].apply(phosphorous_level)
#! Define thresholds for nitrogen levels
def nitrogen_level(n):
    if n > 30:
        return 'High'
    elif 15 <= n <= 30:
        return 'Medium'
    else:
        return 'Low'
data['Nitrogen'] = data['Nitrogen'].apply(nitrogen_level)
#! Define thresholds for potassium levels
def potassium_level(k):
    if k > 30:
        return 'High'
    elif 15 <= k <= 30:
        return 'Medium'
    else:
        return 'Low'
data['Potassium'] = data['Potassium'].apply(potassium_level)
#! Define thresholds for moisture levels
def moisture_level(m):
    if m > 70:
        return 'High'
    elif 40 <= m <= 70:
        return 'Moderate'
    else:
        return 'Low'
data['Moisture'] = data['Moisture'].apply(moisture_level)
#! Define thresholds for temperature levels
def temperature_level(t):
    if t > 30:
        return 'High'
    elif 15 <= t <= 30:
        return 'Medium'
    else:
        return 'Low'
data['Temparature'] = data['Temparature'].apply(temperature_level)
#! Define thresholds for humidity levels
def humidity_level(h):
    if h > 70:
        return 'High'
    elif 40 <= h <= 70:
        return 'Medium'
    else:
        return 'Low'
data['Humidity'] = data['Humidity'].apply(humidity_level)

