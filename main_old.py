"""
Optimized ML module for crop and fertilizer recommendations
Designed for Vercel deployment with performance optimizations
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import pickle
from typing import Dict, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global model cache to avoid reloading
_model_cache = {}

class OptimizedModelTrainer:
    """Optimized model trainer for Vercel deployment"""
    
    def __init__(self):
        self.encoders = {}
        self.feature_columns = None
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load and prepare data with optimized preprocessing"""
        try:
            data_path = os.path.join(BASE_DIR, 'data', 'data.csv')
            data = pd.read_csv(data_path)
            
            # Sample data for faster training (use 20% for demo)
            data = data.sample(frac=0.2, random_state=42).reset_index(drop=True)
            
            # Separate targets
            y_crop = data['Crop Type']
            y_fertilizer = data['Fertilizer Name']
            
            # Prepare features
            X = data.drop(['Crop Type', 'Fertilizer Name'], axis=1)
            
            # Optimized encoding - use LabelEncoder instead of get_dummies
            categorical_cols = ['Soil Type', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
            
            for col in categorical_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.encoders[col] = le
            
            self.feature_columns = X.columns.tolist()
            
            return X, y_crop, y_fertilizer
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def train_lightweight_models(self) -> Tuple[Any, Any, float, float]:
        """Train lightweight models optimized for Vercel"""
        try:
            X, y_crop, y_fertilizer = self.prepare_data()
            
            # Use smaller, faster RandomForest models
            crop_model = RandomForestClassifier(
                n_estimators=50,  # Reduced from 100
                max_depth=15,     # Limit depth
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1  # Single thread for Vercel
            )
            
            fertilizer_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1
            )
            
            # Split data
            X_train, X_test, y_crop_train, y_crop_test = train_test_split(
                X, y_crop, test_size=0.2, random_state=42
            )
            
            _, _, y_fert_train, y_fert_test = train_test_split(
                X, y_fertilizer, test_size=0.2, random_state=42
            )
            
            # Train models
            logger.info("Training crop model...")
            crop_model.fit(X_train, y_crop_train)
            
            logger.info("Training fertilizer model...")
            fertilizer_model.fit(X_train, y_fert_train)
            
            # Evaluate
            crop_pred = crop_model.predict(X_test)
            fert_pred = fertilizer_model.predict(X_test)
            
            crop_accuracy = accuracy_score(y_crop_test, crop_pred)
            fert_accuracy = accuracy_score(y_fert_test, fert_pred)
            
            logger.info(f"Crop model accuracy: {crop_accuracy:.4f}")
            logger.info(f"Fertilizer model accuracy: {fert_accuracy:.4f}")
            
            return crop_model, fertilizer_model, crop_accuracy, fert_accuracy
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def save_models(self, crop_model, fertilizer_model):
        """Save models and encoders"""
        try:
            model_dir = os.path.join(BASE_DIR, 'model')
            os.makedirs(model_dir, exist_ok=True)
            
            # Save models with compression
            joblib.dump(crop_model, os.path.join(model_dir, 'crop_model.pkl'), compress=3)
            joblib.dump(fertilizer_model, os.path.join(model_dir, 'fertilizer_model.pkl'), compress=3)
            
            # Save encoders and feature columns
            with open(os.path.join(model_dir, 'encoders.pkl'), 'wb') as f:
                pickle.dump(self.encoders, f)
            
            with open(os.path.join(model_dir, 'feature_columns.pkl'), 'wb') as f:
                pickle.dump(self.feature_columns, f)
                
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise

def load_model_cache() -> Dict[str, Any]:
    """Load models into cache (singleton pattern for Vercel)"""
    global _model_cache
    
    if _model_cache:
        return _model_cache
    
    try:
        model_dir = os.path.join(BASE_DIR, 'model')
        
        # Load models
        crop_model = joblib.load(os.path.join(model_dir, 'crop_model.pkl'))
        fertilizer_model = joblib.load(os.path.join(model_dir, 'fertilizer_model.pkl'))
        
        # Load encoders and feature columns
        with open(os.path.join(model_dir, 'encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
        
        with open(os.path.join(model_dir, 'feature_columns.pkl'), 'rb') as f:
            feature_columns = pickle.load(f)
        
        _model_cache = {
            'crop_model': crop_model,
            'fertilizer_model': fertilizer_model,
            'encoders': encoders,
            'feature_columns': feature_columns
        }
        
        logger.info("Models loaded into cache")
        return _model_cache
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def encode_input(input_data: Dict[str, Any]) -> np.ndarray:
    """Encode input data using saved encoders"""
    try:
        cache = load_model_cache()
        encoders = cache['encoders']
        feature_columns = cache['feature_columns']
        
        # Create feature vector
        encoded_values = []
        
        for col in feature_columns:
            if col in input_data:
                value = input_data[col]
                if col in encoders:
                    # Handle unseen categories gracefully
                    try:
                        encoded_value = encoders[col].transform([str(value)])[0]
                    except ValueError:
                        # Use most common class if unseen category
                        encoded_value = 0
                    encoded_values.append(encoded_value)
                else:
                    encoded_values.append(float(value))
            else:
                encoded_values.append(0)
        
        return np.array(encoded_values).reshape(1, -1)
        
    except Exception as e:
        logger.error(f"Error encoding input: {e}")
        raise

def predict_crop(soil_type: str, temperature: str, humidity: str, moisture: str,
                nitrogen: str, potassium: str, phosphorus: str) -> Dict[str, Any]:
    """Optimized crop prediction"""
    try:
        cache = load_model_cache()
        model = cache['crop_model']
        
        input_data = {
            'Soil Type': soil_type,
            'Temparature': temperature,
            'Humidity': humidity,
            'Moisture': moisture,
            'Nitrogen': nitrogen,
            'Potassium': potassium,
            'Phosphorous': phosphorus
        }
        
        encoded_input = encode_input(input_data)
        prediction = model.predict(encoded_input)[0]
        
        # Get confidence if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(encoded_input)
            confidence = float(np.max(probabilities) * 100)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'input_data': input_data
        }
        
    except Exception as e:
        logger.error(f"Error in crop prediction: {e}")
        return {
            'prediction': 'Error',
            'confidence': None,
            'error': str(e)
        }

def predict_fertilizer(soil_type: str, temperature: str, humidity: str, moisture: str,
                      nitrogen: str, potassium: str, phosphorus: str) -> Dict[str, Any]:
    """Optimized fertilizer prediction"""
    try:
        cache = load_model_cache()
        model = cache['fertilizer_model']
        
        input_data = {
            'Soil Type': soil_type,
            'Temparature': temperature,
            'Humidity': humidity,
            'Moisture': moisture,
            'Nitrogen': nitrogen,
            'Potassium': potassium,
            'Phosphorous': phosphorus
        }
        
        encoded_input = encode_input(input_data)
        prediction = model.predict(encoded_input)[0]
        
        # Get confidence if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(encoded_input)
            confidence = float(np.max(probabilities) * 100)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'input_data': input_data
        }
        
    except Exception as e:
        logger.error(f"Error in fertilizer prediction: {e}")
        return {
            'prediction': 'Error',
            'confidence': None,
            'error': str(e)
        }

def get_data_sample(limit: int = 100) -> Dict[str, Any]:
    """Get sample data for display"""
    try:
        data_path = os.path.join(BASE_DIR, 'data', 'data.csv')
        data = pd.read_csv(data_path)
        sample = data.sample(n=min(limit, len(data)), random_state=42)
        
        return {
            'data': sample.to_dict('records'),
            'columns': sample.columns.tolist(),
            'total_rows': len(data),
            'sample_size': len(sample)
        }
        
    except Exception as e:
        logger.error(f"Error getting data sample: {e}")
        return {'error': str(e)}

# For model training (run locally)
if __name__ == '__main__':
    trainer = OptimizedModelTrainer()
    crop_model, fertilizer_model, crop_acc, fert_acc = trainer.train_lightweight_models()
    trainer.save_models(crop_model, fertilizer_model)
    
    print(f"Training completed!")
    print(f"Crop accuracy: {crop_acc:.4f}")
    print(f"Fertilizer accuracy: {fert_acc:.4f}")