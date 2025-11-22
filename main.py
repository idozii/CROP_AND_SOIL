"""
Simple ML module for crop and fertilizer recommendations
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_model_cache = {}

def load_model(model_name):
    """Load model with caching"""
    if model_name not in _model_cache:
        model_path = os.path.join(BASE_DIR, 'model', f'{model_name}.pkl')
        _model_cache[model_name] = joblib.load(model_path)
    return _model_cache[model_name]

def train_models():
    """Train crop and fertilizer models"""
    print("Loading data...")
    data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'data.csv'))
    
    # Prepare features and targets
    X = data.drop(['Crop Type', 'Fertilizer Name'], axis=1)
    y_crop = data['Crop Type']
    y_fert = data['Fertilizer Name']
    
    # Encode categorical features
    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Split data
    X_train, X_test, y_crop_train, y_crop_test = train_test_split(
        X, y_crop, test_size=0.2, random_state=42)
    _, _, y_fert_train, y_fert_test = train_test_split(
        X, y_fert, test_size=0.2, random_state=42)
    
    # Train models
    print("Training crop model...")
    crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
    crop_model.fit(X_train, y_crop_train)
    crop_acc = crop_model.score(X_test, y_crop_test)
    
    print("Training fertilizer model...")
    fert_model = RandomForestClassifier(n_estimators=100, random_state=42)
    fert_model.fit(X_train, y_fert_train)
    fert_acc = fert_model.score(X_test, y_fert_test)
    
    # Save models
    os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)
    joblib.dump(crop_model, os.path.join(BASE_DIR, 'model', 'crop_model.pkl'))
    joblib.dump(fert_model, os.path.join(BASE_DIR, 'model', 'fertilizer_model.pkl'))
    joblib.dump(encoders, os.path.join(BASE_DIR, 'model', 'encoders.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(BASE_DIR, 'model', 'features.pkl'))
    
    print(f"\nModels saved!")
    print(f"Crop accuracy: {crop_acc:.4f}")
    print(f"Fertilizer accuracy: {fert_acc:.4f}")

def predict_crop(soil_type, temperature, humidity, moisture,
                 nitrogen, potassium, phosphorus):
    """Predict best crop"""
    try:
        # Load models
        model = load_model('crop_model')
        encoders = load_model('encoders')
        features = load_model('features')
        
        # Convert inputs to proper types
        temperature = float(temperature)
        humidity = float(humidity)
        moisture = float(moisture)
        nitrogen = float(nitrogen)
        potassium = float(potassium)
        phosphorus = float(phosphorus)
        
        # Create input dataframe
        input_data = pd.DataFrame([[
            soil_type, temperature, humidity, moisture,
            nitrogen, potassium, phosphorus
        ]], columns=features)
        
        # Encode categorical features (only soil_type)
        if 'Soil Type' in input_data.columns and 'Soil Type' in encoders:
            input_data['Soil Type'] = encoders['Soil Type'].transform(input_data['Soil Type'])
        
        # Predict
        prediction = model.predict(input_data)[0]
        probas = model.predict_proba(input_data)[0]
        confidence = float(max(probas) * 100)
        
        return {
            'prediction': str(prediction),
            'confidence': round(confidence, 2)
        }
    except Exception as e:
        print(f"Crop prediction error: {e}")
        return {'prediction': f'Error: {str(e)}', 'confidence': 0}

def predict_fertilizer(soil_type, temperature, humidity, moisture,
                       nitrogen, potassium, phosphorus):
    """Predict best fertilizer"""
    try:
        # Load models
        model = load_model('fertilizer_model')
        encoders = load_model('encoders')
        features = load_model('features')
        
        # Convert inputs to proper types
        temperature = float(temperature)
        humidity = float(humidity)
        moisture = float(moisture)
        nitrogen = float(nitrogen)
        potassium = float(potassium)
        phosphorus = float(phosphorus)
        
        # Create input dataframe
        input_data = pd.DataFrame([[
            soil_type, temperature, humidity, moisture,
            nitrogen, potassium, phosphorus
        ]], columns=features)
        
        # Encode categorical features (only soil_type)
        if 'Soil Type' in input_data.columns and 'Soil Type' in encoders:
            input_data['Soil Type'] = encoders['Soil Type'].transform(input_data['Soil Type'])
        
        # Predict
        prediction = model.predict(input_data)[0]
        probas = model.predict_proba(input_data)[0]
        confidence = float(max(probas) * 100)
        
        return {
            'prediction': str(prediction),
            'confidence': round(confidence, 2)
        }
    except Exception as e:
        print(f"Fertilizer prediction error: {e}")
        return {'prediction': f'Error: {str(e)}', 'confidence': 0}

if __name__ == '__main__':
    train_models()
