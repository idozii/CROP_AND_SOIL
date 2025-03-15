import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#! Load the data
data = pd.read_csv('data/data.csv')

# Define features and target for fertilizer recommendation before dropping the column
y_fertilizer = data['Fertilizer Name']

# Drop the 'Fertilizer Name' column as it is not needed for the model
data = data.drop(columns=['Fertilizer Name'])

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

# Encode categorical variables
data_encoded = pd.get_dummies(data, columns=['Soil Type', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'])

# Define features and target for crop recommendation
X_crop = data_encoded.drop('Crop Type', axis=1)
y_crop = data_encoded['Crop Type']

# Define features and target for fertilizer recommendation
X_fertilizer = data_encoded.drop('Crop Type', axis=1)

# Split the data into training and testing sets for crop recommendation
X_crop_train, X_crop_test, y_crop_train, y_crop_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# Split the data into training and testing sets for fertilizer recommendation
X_fertilizer_train, X_fertilizer_test, y_fertilizer_train, y_fertilizer_test = train_test_split(X_fertilizer, y_fertilizer, test_size=0.2, random_state=42)

# Train the RandomForestClassifier for crop recommendation
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_crop_train, y_crop_train)

# Train the RandomForestClassifier for fertilizer recommendation
fertilizer_model = RandomForestClassifier(n_estimators=100, random_state=42)
fertilizer_model.fit(X_fertilizer_train, y_fertilizer_train)

# Evaluate the crop model
y_crop_pred = crop_model.predict(X_crop_test)
print(f'Crop Model Accuracy: {accuracy_score(y_crop_test, y_crop_pred)}')
print(f'Crop Model Classification Report:\n{classification_report(y_crop_test, y_crop_pred)}')
print(f'Crop Model Confusion Matrix:\n{confusion_matrix(y_crop_test, y_crop_pred)}')

# Evaluate the fertilizer model
y_fertilizer_pred = fertilizer_model.predict(X_fertilizer_test)
print(f'Fertilizer Model Accuracy: {accuracy_score(y_fertilizer_test, y_fertilizer_pred)}')
print(f'Fertilizer Model Classification Report:\n{classification_report(y_fertilizer_test, y_fertilizer_pred)}')
print(f'Fertilizer Model Confusion Matrix:\n{confusion_matrix(y_fertilizer_test, y_fertilizer_pred)}')

# Function to recommend crop using the trained model
def recommend_crop_ml(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus):
    input_data = pd.DataFrame({
        'Soil_Type': [soil_type],
        'Temparature': [temperature],
        'Humidity': [humidity],
        'Moisture': [moisture],
        'Nitrogen': [nitrogen],
        'Potassium': [potassium],
        'Phosphorous': [phosphorus]
    })
    input_data_encoded = pd.get_dummies(input_data)
    input_data_encoded = input_data_encoded.reindex(columns=X_crop.columns, fill_value=0)
    prediction = crop_model.predict(input_data_encoded)
    return prediction[0]

# Function to recommend fertilizer using the trained model
def recommend_fertilizer_ml(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus):
    input_data = pd.DataFrame({
        'Soil_Type': [soil_type],
        'Temparature': [temperature],
        'Humidity': [humidity],
        'Moisture': [moisture],
        'Nitrogen': [nitrogen],
        'Potassium': [potassium],
        'Phosphorous': [phosphorus]
    })
    input_data_encoded = pd.get_dummies(input_data)
    input_data_encoded = input_data_encoded.reindex(columns=X_fertilizer.columns, fill_value=0)
    prediction = fertilizer_model.predict(input_data_encoded)
    return prediction[0]

# Example usage with machine learning model
soil_type = 'Sandy'
temperature = 'High'
humidity = 'Medium'
moisture = 'Moderate'
nitrogen = 'Medium'
potassium = 'High'
phosphorus = 'Low'

recommended_crop_ml = recommend_crop_ml(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus)
print(f'Recommended crop (ML): {recommended_crop_ml}')

recommended_fertilizer_ml = recommend_fertilizer_ml(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus)
print(f'Recommended fertilizer (ML): {recommended_fertilizer_ml}')