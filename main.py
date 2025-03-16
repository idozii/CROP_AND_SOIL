import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

#! Load the data
data = pd.read_csv('data/data.csv')

# Define features and target for fertilizer recommendation before dropping the column
fertilizer_label_encoder = LabelEncoder()
y_fertilizer = fertilizer_label_encoder.fit_transform(data['Fertilizer Name'])

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

# Encode the target variable for crop recommendation
crop_label_encoder = LabelEncoder()
y_crop = crop_label_encoder.fit_transform(data_encoded['Crop Type'])

# Define features for crop recommendation
X_crop = data_encoded.drop('Crop Type', axis=1)

# Define features and target for fertilizer recommendation
X_fertilizer = data_encoded.drop('Crop Type', axis=1)

# Split the data into training and testing sets for crop recommendation
X_crop_train, X_crop_test, y_crop_train, y_crop_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# Split the data into training and testing sets for fertilizer recommendation
X_fertilizer_train, X_fertilizer_test, y_fertilizer_train, y_fertilizer_test = train_test_split(X_fertilizer, y_fertilizer, test_size=0.2, random_state=42)

# Train the XGBoost model for crop recommendation
crop_model = xgb.XGBClassifier(tree_method='gpu_hist', random_state=42)
crop_model.fit(X_crop_train, y_crop_train)

# Train the XGBoost model for fertilizer recommendation
fertilizer_model = xgb.XGBClassifier(tree_method='gpu_hist', random_state=42)
fertilizer_model.fit(X_fertilizer_train, y_fertilizer_train)

# Save the trained models
joblib.dump(crop_model, 'crop_model.pkl')
joblib.dump(fertilizer_model, 'fertilizer_model.pkl')
joblib.dump(crop_label_encoder, 'crop_label_encoder.pkl')  # Save the crop label encoder
joblib.dump(fertilizer_label_encoder, 'fertilizer_label_encoder.pkl')  # Save the fertilizer label encoder

# Evaluate the crop model
y_crop_pred = crop_model.predict(X_crop_test)
print(f'Crop Model Accuracy: {accuracy_score(y_crop_test, y_crop_pred)}')
print(f'Crop Model Classification Report:\n{classification_report(y_crop_test, y_crop_pred, target_names=crop_label_encoder.classes_)}')
print(f'Crop Model Confusion Matrix:\n{confusion_matrix(y_crop_test, y_crop_pred)}')

# Evaluate the fertilizer model
y_fertilizer_pred = fertilizer_model.predict(X_fertilizer_test)
print(f'Fertilizer Model Accuracy: {accuracy_score(y_fertilizer_test, y_fertilizer_pred)}')
print(f'Fertilizer Model Classification Report:\n{classification_report(y_fertilizer_test, y_fertilizer_pred, target_names=fertilizer_label_encoder.classes_)}')
print(f'Fertilizer Model Confusion Matrix:\n{confusion_matrix(y_fertilizer_test, y_fertilizer_pred)}')

# Plot confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ConfusionMatrixDisplay.from_estimator(crop_model, X_crop_test, y_crop_test, ax=ax[0], cmap='Blues')
ax[0].set_title('Crop Model Confusion Matrix')

ConfusionMatrixDisplay.from_estimator(fertilizer_model, X_fertilizer_test, y_fertilizer_test, ax=ax[1], cmap='Blues')
ax[1].set_title('Fertilizer Model Confusion Matrix')

plt.tight_layout()
plt.savefig('confusion_matrices.png')  # Save the plot to a file

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
    return crop_label_encoder.inverse_transform(prediction)[0]  # Decode the prediction

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
    return fertilizer_label_encoder.inverse_transform(prediction)[0]  # Decode the prediction

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