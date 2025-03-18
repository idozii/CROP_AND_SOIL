import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import argparse
import os

def train_models(algorithm='rf'):
    algorithms = {
        'rf': ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        'gb': ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        'svm': ('Support Vector Machine', SVC(probability=True, random_state=42)),
        'knn': ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5)),
        'mlp': ('Neural Network', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
        'lr': ('Logistic Regression', LogisticRegression(max_iter=500, random_state=42))
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Algorithm '{algorithm}' not supported. Choose from: {list(algorithms.keys())}")
    
    algo_name, model_class = algorithms[algorithm]
    print(f"Training with {algo_name}...")
    
    data = pd.read_csv('data/data.csv')

    y_fertilizer = data['Fertilizer Name']

    data = data.drop(columns=['Fertilizer Name'])

    numerical_cols = data.select_dtypes(include=[float, int]).columns.tolist()
    categorical_cols = data.select_dtypes(include=[object]).columns.tolist()

    def phosphorous_level(p):
        if p > 30:
            return 'High'
        elif 15 <= p <= 30:
            return 'Medium'
        else:
            return 'Low'
    data['Phosphorous'] = data['Phosphorous'].apply(phosphorous_level)

    def nitrogen_level(n):
        if n > 30:
            return 'High'
        elif 15 <= n <= 30:
            return 'Medium'
        else:
            return 'Low'
    data['Nitrogen'] = data['Nitrogen'].apply(nitrogen_level)

    def potassium_level(k):
        if k > 30:
            return 'High'
        elif 15 <= k <= 30:
            return 'Medium'
        else:
            return 'Low'
    data['Potassium'] = data['Potassium'].apply(potassium_level)

    def moisture_level(m):
        if m > 70:
            return 'High'
        elif 40 <= m <= 70:
            return 'Moderate'
        else:
            return 'Low'
    data['Moisture'] = data['Moisture'].apply(moisture_level)

    def temperature_level(t):
        if t > 30:
            return 'High'
        elif 15 <= t <= 30:
            return 'Medium'
        else:
            return 'Low'
    data['Temparature'] = data['Temparature'].apply(temperature_level)

    def humidity_level(h):
        if h > 70:
            return 'High'
        elif 40 <= h <= 70:
            return 'Medium'
        else:
            return 'Low'
    data['Humidity'] = data['Humidity'].apply(humidity_level)

    data_encoded = pd.get_dummies(data, columns=['Soil Type', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'])

    X_crop = data_encoded.drop('Crop Type', axis=1)
    y_crop = data_encoded['Crop Type']

    X_fertilizer = data_encoded.drop('Crop Type', axis=1)

    X_crop_train, X_crop_test, y_crop_train, y_crop_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

    X_fertilizer_train, X_fertilizer_test, y_fertilizer_train, y_fertilizer_test = train_test_split(X_fertilizer, y_fertilizer, test_size=0.2, random_state=42)

    crop_model = model_class.__class__(**model_class.get_params())
    fertilizer_model = model_class.__class__(**model_class.get_params())

    crop_model.fit(X_crop_train, y_crop_train)

    fertilizer_model.fit(X_fertilizer_train, y_fertilizer_train)

    model_suffix = f"_{algorithm}"
    crop_model_path = f'model/crop_model{model_suffix}.pkl'
    fertilizer_model_path = f'model/fertilizer_model{model_suffix}.pkl'
    
    joblib.dump(crop_model, crop_model_path)
    joblib.dump(fertilizer_model, fertilizer_model_path)
    
    joblib.dump(crop_model, 'model/crop_model.pkl')
    joblib.dump(fertilizer_model, 'model/fertilizer_model.pkl')

    y_crop_pred = crop_model.predict(X_crop_test)
    crop_accuracy = accuracy_score(y_crop_test, y_crop_pred)
    print(f'Crop Model ({algo_name}) Accuracy: {crop_accuracy:.4f}')
    print(f'Crop Model Classification Report:\n{classification_report(y_crop_test, y_crop_pred)}')
    print(f'Crop Model Confusion Matrix:\n{confusion_matrix(y_crop_test, y_crop_pred)}')

    y_fertilizer_pred = fertilizer_model.predict(X_fertilizer_test)
    fertilizer_accuracy = accuracy_score(y_fertilizer_test, y_fertilizer_pred)
    print(f'Fertilizer Model ({algo_name}) Accuracy: {fertilizer_accuracy:.4f}')
    print(f'Fertilizer Model Classification Report:\n{classification_report(y_fertilizer_test, y_fertilizer_pred)}')
    print(f'Fertilizer Model Confusion Matrix:\n{confusion_matrix(y_fertilizer_test, y_fertilizer_pred)}')

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ConfusionMatrixDisplay.from_estimator(crop_model, X_crop_test, y_crop_test, ax=ax[0], cmap='Blues')
    ax[0].set_title(f'Crop Model ({algo_name}) Confusion Matrix')

    ConfusionMatrixDisplay.from_estimator(fertilizer_model, X_fertilizer_test, y_fertilizer_test, ax=ax[1], cmap='Blues')
    ax[1].set_title(f'Fertilizer Model ({algo_name}) Confusion Matrix')

    plt.tight_layout()
    plt.savefig(f'static/figures/confusion_matrices_{algorithm}.png')
    
    return crop_model, fertilizer_model, crop_accuracy, fertilizer_accuracy

def recommend_crop_ml(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus, model=None):
    if model is None:
        model = joblib.load('model/crop_model.pkl')
        
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
    
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        data = pd.read_csv('data/data.csv')
        data = data.drop(columns=['Fertilizer Name'])
        data_encoded = pd.get_dummies(data, columns=['Soil Type', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'])
        feature_names = data_encoded.drop('Crop Type', axis=1).columns
        
    input_data_encoded = input_data_encoded.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(input_data_encoded)
    
    # Get probability scores if model supports it
    probabilities = None
    confidence = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data_encoded)
        confidence = np.max(probabilities) * 100  # Convert to percentage
    
    # Get feature importances if possible
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        # Sort by importance (descending)
        feature_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                     key=lambda item: item[1], 
                                                     reverse=True)[:5]}  # Top 5 features
    
    # Return prediction with additional information
    result = {
        'prediction': prediction[0],
        'confidence': confidence,
        'probabilities': probabilities,
        'top_features': feature_importance
    }
    
    return result


def recommend_fertilizer_ml(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus, model=None):
    if model is None:
        model = joblib.load('model/fertilizer_model.pkl')
        
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
    
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        data = pd.read_csv('data/data.csv')
        data = data.drop(columns=['Fertilizer Name'])
        data_encoded = pd.get_dummies(data, columns=['Soil Type', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'])
        feature_names = data_encoded.drop('Crop Type', axis=1).columns
        
    input_data_encoded = input_data_encoded.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(input_data_encoded)
    
    # Get probability scores if model supports it
    probabilities = None
    confidence = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data_encoded)
        confidence = np.max(probabilities) * 100  # Convert to percentage
    
    # Get feature importances if possible
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        # Sort by importance (descending)
        feature_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                     key=lambda item: item[1], 
                                                     reverse=True)[:5]}  # Top 5 features
    
    # Return prediction with additional information
    result = {
        'prediction': prediction[0],
        'confidence': confidence,
        'probabilities': probabilities,
        'top_features': feature_importance
    }
    
    return result


def ensemble_predict(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus):
    algorithms = ['rf', 'gb', 'svm', 'knn']
    
    # Track votes for each class
    crop_votes = {}
    fertilizer_votes = {}
    
    # Track probabilities if available
    crop_probas = {}
    fertilizer_probas = {}
    
    # Load and predict with each model
    for algo in algorithms:
        try:
            # Load models if they exist
            crop_model_path = f'model/crop_model_{algo}.pkl'
            fert_model_path = f'model/fertilizer_model_{algo}.pkl'
            
            if os.path.exists(crop_model_path):
                crop_model = joblib.load(crop_model_path)
                crop_result = recommend_crop_ml(
                    soil_type, temperature, humidity, moisture,
                    nitrogen, potassium, phosphorus, crop_model
                )
                
                # Add to votes
                crop_pred = crop_result['prediction']
                crop_votes[crop_pred] = crop_votes.get(crop_pred, 0) + 1
                
                # Add to probabilities if available
                if crop_result['probabilities'] is not None:
                    probas = crop_result['probabilities'][0]
                    
                    # Get class names from model
                    if hasattr(crop_model, 'classes_'):
                        classes = crop_model.classes_
                        for i, c in enumerate(classes):
                            crop_probas[c] = crop_probas.get(c, 0) + probas[i]
            
            if os.path.exists(fert_model_path):
                fert_model = joblib.load(fert_model_path)
                fert_result = recommend_fertilizer_ml(
                    soil_type, temperature, humidity, moisture,
                    nitrogen, potassium, phosphorus, fert_model
                )
                
                # Add to votes
                fert_pred = fert_result['prediction']
                fertilizer_votes[fert_pred] = fertilizer_votes.get(fert_pred, 0) + 1
                
                # Add to probabilities if available
                if fert_result['probabilities'] is not None:
                    probas = fert_result['probabilities'][0]
                    
                    # Get class names from model
                    if hasattr(fert_model, 'classes_'):
                        classes = fert_model.classes_
                        for i, c in enumerate(classes):
                            fertilizer_probas[c] = fertilizer_probas.get(c, 0) + probas[i]
                
        except Exception as e:
            print(f"Error with {algo} model: {str(e)}")
            continue
    
    # Calculate weighted predictions based on votes and probabilities
    
    # For crops
    crop_prediction = None
    crop_confidence = 0
    if crop_votes:
        # Get crop with most votes
        crop_prediction = max(crop_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence
        if crop_probas:
            # Normalize probabilities
            total = sum(crop_probas.values())
            if total > 0:
                for crop in crop_probas:
                    crop_probas[crop] /= total
                
                # Get confidence for predicted crop
                crop_confidence = crop_probas.get(crop_prediction, 0) * 100
            else:
                # If no probabilities, use vote percentage
                crop_confidence = (crop_votes[crop_prediction] / sum(crop_votes.values())) * 100
    
    # For fertilizers
    fertilizer_prediction = None
    fertilizer_confidence = 0
    if fertilizer_votes:
        # Get fertilizer with most votes
        fertilizer_prediction = max(fertilizer_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence
        if fertilizer_probas:
            # Normalize probabilities
            total = sum(fertilizer_probas.values())
            if total > 0:
                for fert in fertilizer_probas:
                    fertilizer_probas[fert] /= total
                
                # Get confidence for predicted fertilizer
                fertilizer_confidence = fertilizer_probas.get(fertilizer_prediction, 0) * 100
            else:
                # If no probabilities, use vote percentage
                fertilizer_confidence = (fertilizer_votes[fertilizer_prediction] / sum(fertilizer_votes.values())) * 100
    
    # Get top alternatives (second best options)
    crop_alternatives = []
    if len(crop_votes) > 1:
        # Sort crops by votes, descending
        sorted_crops = sorted(crop_votes.items(), key=lambda x: x[1], reverse=True)
        # Take second best
        crop_alternatives = [sorted_crops[1][0]]
    
    fertilizer_alternatives = []
    if len(fertilizer_votes) > 1:
        # Sort fertilizers by votes, descending
        sorted_ferts = sorted(fertilizer_votes.items(), key=lambda x: x[1], reverse=True)
        # Take second best
        fertilizer_alternatives = [sorted_ferts[1][0]]
    
    # Prepare and return results
    results = {
        'crop_prediction': crop_prediction,
        'crop_confidence': round(crop_confidence, 2),
        'crop_votes': crop_votes,
        'crop_alternatives': crop_alternatives,
        
        'fertilizer_prediction': fertilizer_prediction,
        'fertilizer_confidence': round(fertilizer_confidence, 2),
        'fertilizer_votes': fertilizer_votes,
        'fertilizer_alternatives': fertilizer_alternatives,
        
        'models_used': [algo for algo in algorithms 
                       if os.path.exists(f'model/crop_model_{algo}.pkl') 
                       or os.path.exists(f'model/fertilizer_model_{algo}.pkl')]
    }
    
    return results


def analyze_features_impact(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus):
    """Analyze how each feature impacts the predictions
    
    Returns a report showing how changing each feature affects predictions
    """
    # Load the best model (assume it's RandomForest)
    try:
        crop_model = joblib.load('model/crop_model_rf.pkl')
        fert_model = joblib.load('model/fertilizer_model_rf.pkl')
    except:
        # Fallback to default models
        crop_model = joblib.load('model/crop_model.pkl')
        fert_model = joblib.load('model/fertilizer_model.pkl')
    
    # Make baseline prediction
    baseline_crop = recommend_crop_ml(
        soil_type, temperature, humidity, moisture,
        nitrogen, potassium, phosphorus, crop_model
    )
    
    baseline_fert = recommend_fertilizer_ml(
        soil_type, temperature, humidity, moisture,
        nitrogen, potassium, phosphorus, fert_model
    )
    
    # Define alternative values for each feature
    alternatives = {
        'soil_type': ['Sandy', 'Loamy', 'Clayey', 'Black'],
        'temperature': ['Low', 'Medium', 'High'],
        'humidity': ['Low', 'Medium', 'High'],
        'moisture': ['Low', 'Moderate', 'High'],
        'nitrogen': ['Low', 'Medium', 'High'],
        'potassium': ['Low', 'Medium', 'High'],
        'phosphorus': ['Low', 'Medium', 'High']
    }
    
    # Store results
    impact_analysis = {}
    
    # For each feature, try different values and track changes
    for feature, values in alternatives.items():
        feature_results = []
        
        for val in values:
            if val == locals()[feature]:  # Skip current value
                continue
                
            # Create a copy of parameters with this value changed
            kwargs = {
                'soil_type': soil_type,
                'temperature': temperature,
                'humidity': humidity,
                'moisture': moisture,
                'nitrogen': nitrogen,
                'potassium': potassium,
                'phosphorus': phosphorus,
                'model': crop_model  # Use same model
            }
            
            # Update feature value
            kwargs[feature] = val
            
            # Make prediction with changed feature
            crop_result = recommend_crop_ml(**kwargs)
            
            # Check if prediction changed
            changed = crop_result['prediction'] != baseline_crop['prediction']
            
            # Update to use fertilizer model
            kwargs['model'] = fert_model
            fert_result = recommend_fertilizer_ml(**kwargs)
            changed_fert = fert_result['prediction'] != baseline_fert['prediction']
            
            # Record results
            feature_results.append({
                'value': val,
                'crop_prediction': crop_result['prediction'],
                'crop_changed': changed,
                'crop_confidence': crop_result['confidence'],
                'fertilizer_prediction': fert_result['prediction'],
                'fertilizer_changed': changed_fert,
                'fertilizer_confidence': fert_result['confidence']
            })
        
        impact_analysis[feature] = feature_results
    
    # Add baseline predictions
    baseline = {
        'crop_prediction': baseline_crop['prediction'],
        'crop_confidence': baseline_crop['confidence'],
        'fertilizer_prediction': baseline_fert['prediction'],
        'fertilizer_confidence': baseline_fert['confidence'],
        'soil_type': soil_type,
        'temperature': temperature,
        'humidity': humidity,
        'moisture': moisture,
        'nitrogen': nitrogen,
        'potassium': potassium,
        'phosphorus': phosphorus
    }
    
    return {
        'baseline': baseline,
        'impact': impact_analysis
    }


def get_similar_inputs(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus, limit=5):
    """Find similar inputs from the training data
    
    Returns rows from training data that are most similar to the given input
    """
    try:
        data = pd.read_csv('data/data.csv')
        
        # Apply same preprocessing as in train_models
        for col, func, mapping in [
            ('Phosphorous', lambda p: 'High' if p > 30 else ('Medium' if 15 <= p <= 30 else 'Low'), 
                {'High': 3, 'Medium': 2, 'Low': 1}),
            ('Nitrogen', lambda n: 'High' if n > 30 else ('Medium' if 15 <= n <= 30 else 'Low'),
                {'High': 3, 'Medium': 2, 'Low': 1}),
            ('Potassium', lambda k: 'High' if k > 30 else ('Medium' if 15 <= k <= 30 else 'Low'),
                {'High': 3, 'Medium': 2, 'Low': 1}),
            ('Moisture', lambda m: 'High' if m > 70 else ('Moderate' if 40 <= m <= 70 else 'Low'),
                {'High': 3, 'Moderate': 2, 'Low': 1}),
            ('Temparature', lambda t: 'High' if t > 30 else ('Medium' if 15 <= t <= 30 else 'Low'),
                {'High': 3, 'Medium': 2, 'Low': 1}),
            ('Humidity', lambda h: 'High' if h > 70 else ('Medium' if 40 <= h <= 70 else 'Low'),
                {'High': 3, 'Medium': 2, 'Low': 1})
        ]:
            # Ensure consistent types by applying preprocessing to numeric data
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].apply(func)
            
            # Map to numeric for distance calculation
            data[f'{col}_num'] = data[col].map(mapping)
        
        # Create mapping for soil type
        soil_types = {s: i+1 for i, s in enumerate(data['Soil Type'].unique())}
        data['Soil_Type_num'] = data['Soil Type'].map(soil_types)
        
        # Create numeric input vector
        input_dict = {
            'Soil_Type_num': soil_types.get(soil_type, 1),
            'Temparature_num': {'High': 3, 'Medium': 2, 'Low': 1}.get(temperature, 2),
            'Humidity_num': {'High': 3, 'Medium': 2, 'Low': 1}.get(humidity, 2),
            'Moisture_num': {'High': 3, 'Moderate': 2, 'Low': 1}.get(moisture, 2),
            'Nitrogen_num': {'High': 3, 'Medium': 2, 'Low': 1}.get(nitrogen, 2),
            'Potassium_num': {'High': 3, 'Medium': 2, 'Low': 1}.get(potassium, 2),
            'Phosphorous_num': {'High': 3, 'Medium': 2, 'Low': 1}.get(phosphorus, 2)
        }
        
        # Calculate Euclidean distance
        data['distance'] = np.sqrt(
            (data['Soil_Type_num'] - input_dict['Soil_Type_num']) ** 2 +
            (data['Temparature_num'] - input_dict['Temparature_num']) ** 2 +
            (data['Humidity_num'] - input_dict['Humidity_num']) ** 2 +
            (data['Moisture_num'] - input_dict['Moisture_num']) ** 2 +
            (data['Nitrogen_num'] - input_dict['Nitrogen_num']) ** 2 +
            (data['Potassium_num'] - input_dict['Potassium_num']) ** 2 +
            (data['Phosphorous_num'] - input_dict['Phosphorous_num']) ** 2
        )
        
        # Get closest matches
        similar_rows = data.sort_values('distance').head(limit)[
            ['Soil Type', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 
             'Potassium', 'Phosphorous', 'Crop Type', 'Fertilizer Name', 'distance']
        ].to_dict('records')
        
        return similar_rows
    
    except Exception as e:
        print(f"Error finding similar inputs: {str(e)}")
        return None

def compare_algorithms():
    algorithms = ['rf', 'gb', 'svm', 'knn', 'mlp', 'lr']
    results = []
    
    for algo in algorithms:
        try:
            print(f"\n--- Training {algo} models ---")
            _, _, crop_acc, fert_acc = train_models(algo)
            results.append({
                'Algorithm': algo,
                'Crop Accuracy': crop_acc,
                'Fertilizer Accuracy': fert_acc,
                'Average Accuracy': (crop_acc + fert_acc) / 2
            })
        except Exception as e:
            print(f"Error with algorithm {algo}: {str(e)}")
    
    results_df = pd.DataFrame(results).sort_values('Average Accuracy', ascending=False)
    print("\n--- Algorithm Comparison Results ---")
    print(results_df)
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(results_df))
    width = 0.35
    
    plt.bar(x - width/2, results_df['Crop Accuracy'], width, label='Crop Accuracy')
    plt.bar(x + width/2, results_df['Fertilizer Accuracy'], width, label='Fertilizer Accuracy')
    
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison by Algorithm')
    plt.xticks(x, results_df['Algorithm'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/figures/algorithm_comparison.png')
    print("Comparison chart saved to static/figures/algorithm_comparison.png")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Advanced ML for crop and fertilizer recommendation models')
    parser.add_argument('--algorithm', '-a', choices=['rf', 'gb', 'svm', 'knn', 'mlp', 'lr', 'ensemble'], 
                        default='rf', help='Machine learning algorithm to use')
    parser.add_argument('--compare', '-c', action='store_true', 
                        help='Compare all algorithms')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run a test prediction with the trained model')
    parser.add_argument('--analyze', '-an', action='store_true',
                        help='Analyze feature impact')
    parser.add_argument('--similar', '-s', type=int, default=0,
                        help='Find N similar inputs from training data')
    parser.add_argument('--ensemble', '-e', action='store_true',
                        help='Create and test ensemble model')
    
    args = parser.parse_args()
    
    test_values = {
        'soil_type': 'Sandy',
        'temperature': 'High',
        'humidity': 'Medium',
        'moisture': 'Moderate',
        'nitrogen': 'Medium',
        'potassium': 'High',
        'phosphorus': 'Low'
    }
    
    os.makedirs('static/figures', exist_ok=True)
    
    if args.compare:
        print("Comparing all algorithms...")
        results = compare_algorithms()
        print("\nBest algorithm based on average accuracy:")
        best_algo = results.iloc[0]['Algorithm']
        print(f"  {best_algo} (Crop: {results.iloc[0]['Crop Accuracy']:.4f}, " 
              f"Fertilizer: {results.iloc[0]['Fertilizer Accuracy']:.4f})")
    
    elif args.ensemble:
        print("Creating ensemble model...")
        algorithms = ['rf', 'gb', 'svm', 'knn']
        for algo in algorithms:
            if not os.path.exists(f'model/crop_model_{algo}.pkl'):
                print(f"Training {algo} model first...")
                train_models(algo)
        
        print("\n--- Ensemble Model Test ---")
        ensemble_result = ensemble_predict(**test_values)
        
        print(f"Ensemble prediction from {len(ensemble_result['models_used'])} models:")
        print(f"  Recommended crop: {ensemble_result['crop_prediction']} "
              f"(confidence: {ensemble_result['crop_confidence']}%)")
        if ensemble_result['crop_alternatives']:
            print(f"  Alternative crop: {ensemble_result['crop_alternatives'][0]}")
        
        print(f"  Recommended fertilizer: {ensemble_result['fertilizer_prediction']} "
              f"(confidence: {ensemble_result['fertilizer_confidence']}%)")
        if ensemble_result['fertilizer_alternatives']:
            print(f"  Alternative fertilizer: {ensemble_result['fertilizer_alternatives'][0]}")
            
        print(f"\nModels used in ensemble: {', '.join(ensemble_result['models_used'])}")
        
    else:
        if args.algorithm == 'ensemble':
            print("For ensemble predictions, use --ensemble/-e flag instead")
            return
            
        print(f"Training {args.algorithm} model...")
        crop_model, fertilizer_model, crop_acc, fert_acc = train_models(args.algorithm)
        
        print(f"\n--- Model Training Results ---")
        print(f"Crop model accuracy: {crop_acc:.4f}")
        print(f"Fertilizer model accuracy: {fert_acc:.4f}")
        
        if args.test:
            print("\n--- Test Prediction Results ---")
            print(f"Input values:")
            for k, v in test_values.items():
                print(f"  {k.capitalize()}: {v}")
            
            recommended_crop = recommend_crop_ml(
                model=crop_model,
                **test_values
            )
            
            recommended_fertilizer = recommend_fertilizer_ml(
                model=fertilizer_model,
                **test_values
            )
            
            print(f"\nPredictions:")
            print(f"  Recommended crop: {recommended_crop['prediction']}")
            if recommended_crop['confidence']:
                print(f"  Crop confidence: {recommended_crop['confidence']:.2f}%")
            
            print(f"  Recommended fertilizer: {recommended_fertilizer['prediction']}")
            if recommended_fertilizer['confidence']:
                print(f"  Fertilizer confidence: {recommended_fertilizer['confidence']:.2f}%")
            
            if recommended_crop['top_features']:
                print("\nTop features for crop recommendation:")
                for feature, importance in recommended_crop['top_features'].items():
                    print(f"  {feature}: {importance:.4f}")
            
            if recommended_fertilizer['top_features']:
                print("\nTop features for fertilizer recommendation:")
                for feature, importance in recommended_fertilizer['top_features'].items():
                    print(f"  {feature}: {importance:.4f}")
        
        if args.analyze:
            print("\n--- Feature Impact Analysis ---")
            impact_results = analyze_features_impact(**test_values)
            
            print("Baseline prediction:")
            print(f"  Crop: {impact_results['baseline']['crop_prediction']}")
            print(f"  Fertilizer: {impact_results['baseline']['fertilizer_prediction']}")
            
            print("\nFeature impacts that change predictions:")
            for feature, results in impact_results['impact'].items():
                changed_crop = [r for r in results if r['crop_changed']]
                changed_fert = [r for r in results if r['fertilizer_changed']]
                
                if changed_crop:
                    print(f"  {feature.capitalize()}: Changes crop prediction when set to: "
                          f"{', '.join([r['value'] + ' → ' + r['crop_prediction'] for r in changed_crop])}")
                
                if changed_fert:
                    print(f"  {feature.capitalize()}: Changes fertilizer prediction when set to: "
                          f"{', '.join([r['value'] + ' → ' + r['fertilizer_prediction'] for r in changed_fert])}")
                    
                if not changed_crop and not changed_fert:
                    print(f"  {feature.capitalize()}: No significant impact on prediction")
        
        if args.similar > 0:
            print(f"\n--- Finding {args.similar} Similar Inputs from Training Data ---")
            similar_inputs = get_similar_inputs(**test_values, limit=args.similar)
            
            if similar_inputs:
                print(f"{'Soil Type':<10} {'Temp':<6} {'Humidity':<8} {'Moisture':<8} {'N':<3} {'K':<3} {'P':<3} {'Crop':<15} {'Fertilizer':<15}")
                print('-' * 80)
                for idx, row in enumerate(similar_inputs):
                    print(f"{row['Soil Type']:<10} {row['Temparature']:<6} {row['Humidity']:<8} "
                          f"{row['Moisture']:<8} {row['Nitrogen']:<3} {row['Potassium']:<3} {row['Phosphorous']:<3} "
                          f"{row['Crop Type']:<15} {row['Fertilizer Name']:<15}")


if __name__ == '__main__':
    main()