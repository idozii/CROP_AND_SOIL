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

def train_models(algorithm='rf'):
    """Train crop and fertilizer models using the specified algorithm"""
    # Define algorithm mapping
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

    # Create copies of the model class for crop and fertilizer
    crop_model = model_class.__class__(**model_class.get_params())
    fertilizer_model = model_class.__class__(**model_class.get_params())

    # Train for crop recommendation
    crop_model.fit(X_crop_train, y_crop_train)

    # Train for fertilizer recommendation
    fertilizer_model.fit(X_fertilizer_train, y_fertilizer_train)

    # Save models with algorithm name in filename
    model_suffix = f"_{algorithm}"
    crop_model_path = f'model/crop_model{model_suffix}.pkl'
    fertilizer_model_path = f'model/fertilizer_model{model_suffix}.pkl'
    
    joblib.dump(crop_model, crop_model_path)
    joblib.dump(fertilizer_model, fertilizer_model_path)
    
    # Also save as default models for app.py
    joblib.dump(crop_model, 'model/crop_model.pkl')
    joblib.dump(fertilizer_model, 'model/fertilizer_model.pkl')

    # Evaluate the crop model
    y_crop_pred = crop_model.predict(X_crop_test)
    crop_accuracy = accuracy_score(y_crop_test, y_crop_pred)
    print(f'Crop Model ({algo_name}) Accuracy: {crop_accuracy:.4f}')
    print(f'Crop Model Classification Report:\n{classification_report(y_crop_test, y_crop_pred)}')
    print(f'Crop Model Confusion Matrix:\n{confusion_matrix(y_crop_test, y_crop_pred)}')

    # Evaluate the fertilizer model
    y_fertilizer_pred = fertilizer_model.predict(X_fertilizer_test)
    fertilizer_accuracy = accuracy_score(y_fertilizer_test, y_fertilizer_pred)
    print(f'Fertilizer Model ({algo_name}) Accuracy: {fertilizer_accuracy:.4f}')
    print(f'Fertilizer Model Classification Report:\n{classification_report(y_fertilizer_test, y_fertilizer_pred)}')
    print(f'Fertilizer Model Confusion Matrix:\n{confusion_matrix(y_fertilizer_test, y_fertilizer_pred)}')

    # Plot confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ConfusionMatrixDisplay.from_estimator(crop_model, X_crop_test, y_crop_test, ax=ax[0], cmap='Blues')
    ax[0].set_title(f'Crop Model ({algo_name}) Confusion Matrix')

    ConfusionMatrixDisplay.from_estimator(fertilizer_model, X_fertilizer_test, y_fertilizer_test, ax=ax[1], cmap='Blues')
    ax[1].set_title(f'Fertilizer Model ({algo_name}) Confusion Matrix')

    plt.tight_layout()
    plt.savefig(f'static/figures/confusion_matrices_{algorithm}.png')
    
    return crop_model, fertilizer_model, crop_accuracy, fertilizer_accuracy


def recommend_crop_ml(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus, model=None):
    """Recommend crop using ML model"""
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
    
    # Get the feature names from the model
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        # Use this for older scikit-learn versions
        data = pd.read_csv('data/data.csv')
        data = data.drop(columns=['Fertilizer Name'])
        data_encoded = pd.get_dummies(data, columns=['Soil Type', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'])
        feature_names = data_encoded.drop('Crop Type', axis=1).columns
        
    input_data_encoded = input_data_encoded.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(input_data_encoded)
    return prediction[0]


def recommend_fertilizer_ml(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus, model=None):
    """Recommend fertilizer using ML model"""
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
    
    # Get the feature names from the model
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        # Use this for older scikit-learn versions
        data = pd.read_csv('data/data.csv')
        data = data.drop(columns=['Fertilizer Name'])
        data_encoded = pd.get_dummies(data, columns=['Soil Type', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'])
        feature_names = data_encoded.drop('Crop Type', axis=1).columns
        
    input_data_encoded = input_data_encoded.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(input_data_encoded)
    return prediction[0]


def compare_algorithms():
    """Train and compare all algorithms"""
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
    
    # Create results DataFrame and sort by average accuracy
    results_df = pd.DataFrame(results).sort_values('Average Accuracy', ascending=False)
    print("\n--- Algorithm Comparison Results ---")
    print(results_df)
    
    # Plot results
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train crop and fertilizer recommendation models')
    parser.add_argument('--algorithm', '-a', choices=['rf', 'gb', 'svm', 'knn', 'mlp', 'lr'], 
                        default='rf', help='Machine learning algorithm to use')
    parser.add_argument('--compare', '-c', action='store_true', 
                        help='Compare all algorithms')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run a test prediction with the trained model')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_algorithms()
    else:
        crop_model, fertilizer_model, _, _ = train_models(args.algorithm)
        
        if args.test:
            # Example usage with machine learning model
            soil_type = 'Sandy'
            temperature = 'High'
            humidity = 'Medium'
            moisture = 'Moderate'
            nitrogen = 'Medium'
            potassium = 'High'
            phosphorus = 'Low'

            recommended_crop_ml = recommend_crop_ml(
                soil_type, temperature, humidity, moisture, 
                nitrogen, potassium, phosphorus, crop_model
            )
            recommended_fertilizer_ml = recommend_fertilizer_ml(
                soil_type, temperature, humidity, moisture, 
                nitrogen, potassium, phosphorus, fertilizer_model
            )
            
            print("\n--- Test Prediction Results ---")
            print(f"Input values:")
            print(f"  Soil Type: {soil_type}")
            print(f"  Temperature: {temperature}")
            print(f"  Humidity: {humidity}")
            print(f"  Moisture: {moisture}")
            print(f"  Nitrogen: {nitrogen}")
            print(f"  Potassium: {potassium}")
            print(f"  Phosphorus: {phosphorus}")
            print(f"\nPredictions:")
            print(f"  Recommended crop: {recommended_crop_ml}")
            print(f"  Recommended fertilizer: {recommended_fertilizer_ml}")