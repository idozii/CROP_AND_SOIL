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
    
    print(f"Dataset size: {len(data)} records")
    print(f"Unique crops: {data['Crop Type'].nunique()}")
    print(f"Unique fertilizers: {data['Fertilizer Name'].nunique()}")

    X = data[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
    y_crop = data['Crop Type']
    y_fert = data['Fertilizer Name']
 
    soil_encoder = LabelEncoder()
    X_encoded = X.copy()
    X_encoded['Soil Type'] = soil_encoder.fit_transform(X['Soil Type'])

    X_train, X_test, y_crop_train, y_crop_test = train_test_split(
        X_encoded, y_crop, test_size=0.2, random_state=42, stratify=y_crop)
    X_train_fert, X_test_fert, y_fert_train, y_fert_test = train_test_split(
        X_encoded, y_fert, test_size=0.2, random_state=42, stratify=y_fert)

    crop_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    crop_model.fit(X_train, y_crop_train)
    crop_acc = crop_model.score(X_test, y_crop_test)

    fert_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    fert_model.fit(X_train_fert, y_fert_train)
    fert_acc = fert_model.score(X_test_fert, y_fert_test)
    
    os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)
    joblib.dump(crop_model, os.path.join(BASE_DIR, 'model', 'crop_model.pkl'), compress=9)
    joblib.dump(fert_model, os.path.join(BASE_DIR, 'model', 'fertilizer_model.pkl'), compress=9)
    joblib.dump(soil_encoder, os.path.join(BASE_DIR, 'model', 'soil_encoder.pkl'), compress=9)

    print(f"✅ Models saved successfully!")
    print(f"📊 Crop model accuracy: {crop_acc:.2%}")
    print(f"📊 Fertilizer model accuracy: {fert_acc:.2%}")

def predict_crop(soil_type, temperature, humidity, moisture,
                 nitrogen, potassium, phosphorus):
    """Predict best crop"""
    try:
        model = load_model('crop_model')
        soil_encoder = load_model('soil_encoder')

        temp = float(temperature)
        hum = float(humidity)
        mois = float(moisture)
        n = float(nitrogen)
        k = float(potassium)
        p = float(phosphorus)

        soil_encoded = soil_encoder.transform([soil_type])[0]

        input_data = np.array([[temp, hum, mois, soil_encoded, n, k, p]])
        prediction = model.predict(input_data)[0]
        probas = model.predict_proba(input_data)[0]
        confidence = float(max(probas) * 100)
        
        return {
            'prediction': str(prediction),
            'confidence': round(confidence, 2)
        }
    except Exception as e:
        print(f"Crop prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {'prediction': f'Error: {str(e)}', 'confidence': 0}

def predict_fertilizer(soil_type, temperature, humidity, moisture,
                       nitrogen, potassium, phosphorus):
    """Predict best fertilizer"""
    try:
        model = load_model('fertilizer_model')
        soil_encoder = load_model('soil_encoder')
    
        temp = float(temperature)
        hum = float(humidity)
        mois = float(moisture)
        n = float(nitrogen)
        k = float(potassium)
        p = float(phosphorus)
        
        soil_encoded = soil_encoder.transform([soil_type])[0]
        
        input_data = np.array([[temp, hum, mois, soil_encoded, n, k, p]])
        prediction = model.predict(input_data)[0]
        probas = model.predict_proba(input_data)[0]
        confidence = float(max(probas) * 100)
        
        return {
            'prediction': str(prediction),
            'confidence': round(confidence, 2)
        }
    except Exception as e:
        print(f"Fertilizer prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {'prediction': f'Error: {str(e)}', 'confidence': 0}

if __name__ == '__main__':
    train_models()
