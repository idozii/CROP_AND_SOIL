from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'agriculture_secret_key'

# Default models
crop_model = joblib.load('model/crop_model.pkl')
fertilizer_model = joblib.load('model/fertilizer_model.pkl')
current_algorithm = 'rf'  # Default algorithm

# Define value transformations to match main.py preprocessing
def preprocess_input(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus):
    """Transform numerical values to categories as done in main.py"""
    
    # Convert string values to float if they are numbers
    try:
        temperature = float(temperature)
        humidity = float(humidity)
        moisture = float(moisture)
        nitrogen = float(nitrogen)
        potassium = float(potassium)
        phosphorus = float(phosphorus)
    except ValueError:
        # If already categorical, leave as is
        pass
    
    # Apply transformations based on main.py thresholds
    if isinstance(temperature, (int, float)):
        if temperature > 30:
            temperature = 'High'
        elif 15 <= temperature <= 30:
            temperature = 'Medium'
        else:
            temperature = 'Low'
            
    if isinstance(humidity, (int, float)):
        if humidity > 70:
            humidity = 'High'
        elif 40 <= humidity <= 70:
            humidity = 'Medium'
        else:
            humidity = 'Low'
            
    if isinstance(moisture, (int, float)):
        if moisture > 70:
            moisture = 'High'
        elif 40 <= moisture <= 70:
            moisture = 'Moderate'
        else:
            moisture = 'Low'
            
    if isinstance(nitrogen, (int, float)):
        if nitrogen > 30:
            nitrogen = 'High'
        elif 15 <= nitrogen <= 30:
            nitrogen = 'Medium'
        else:
            nitrogen = 'Low'
            
    if isinstance(potassium, (int, float)):
        if potassium > 30:
            potassium = 'High'
        elif 15 <= potassium <= 30:
            potassium = 'Medium'
        else:
            potassium = 'Low'
            
    if isinstance(phosphorus, (int, float)):
        if phosphorus > 30:
            phosphorus = 'High'
        elif 15 <= phosphorus <= 30:
            phosphorus = 'Medium'
        else:
            phosphorus = 'Low'
            
    return soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')  # Changed from username to email to match the form
        password = request.form.get('password')
        # Store user info in session (in a real app, you'd validate credentials)
        session['user_email'] = email
        return redirect(url_for('user'))
    return render_template('login.html')

@app.route('/user')
def user():
    return render_template('user.html')

@app.route('/real_data')
def real_data():
    try:
        data = pd.read_csv('data/data.csv')
        data_html = data.to_html(
            classes='table table-striped table-hover',
            index=False, 
            border=0,    
            justify='left'
        )
        return render_template('real_data.html', tables=[data_html], titles=data.columns.values)
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        return render_template('real_data.html', error=error_msg)

@app.route('/recommendations')
def recommendations():
    # Show a different view if someone navigates directly to the URL
    return render_template('recommendations_landing.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/select_algorithm', methods=['GET', 'POST'])
def select_algorithm():
    global crop_model, fertilizer_model, current_algorithm
    
    available_algorithms = {
        'rf': 'Random Forest',
        'gb': 'Gradient Boosting',
        'svm': 'Support Vector Machine',
        'knn': 'K-Nearest Neighbors',
        'mlp': 'Neural Network',
        'lr': 'Logistic Regression'
    }
    
    if request.method == 'POST':
        algorithm = request.form.get('algorithm')
        if algorithm in available_algorithms:
            try:
                # Load the selected models
                crop_model = joblib.load(f'model/crop_model_{algorithm}.pkl')
                fertilizer_model = joblib.load(f'model/fertilizer_model_{algorithm}.pkl')
                current_algorithm = algorithm
                return render_template('select_algorithm.html', 
                                      algorithms=available_algorithms,
                                      current=algorithm,
                                      message=f"Successfully loaded {available_algorithms[algorithm]} models")
            except Exception as e:
                return render_template('select_algorithm.html', 
                                      algorithms=available_algorithms,
                                      current=current_algorithm,
                                      error=f"Error loading model: {str(e)}")
    
    # For GET request or after processing POST
    return render_template('select_algorithm.html', 
                          algorithms=available_algorithms,
                          current=current_algorithm)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        soil_type = request.form['soil_type']
        temperature = request.form['temperature'] 
        humidity = request.form['humidity']
        moisture = request.form['moisture']
        nitrogen = request.form['nitrogen']
        potassium = request.form['potassium']
        phosphorus = request.form['phosphorus']
        
        # Store original values for display
        original_values = {
            'soil_type': soil_type,
            'temperature': temperature,
            'humidity': humidity,
            'moisture': moisture,
            'nitrogen': nitrogen,
            'potassium': potassium,
            'phosphorus': phosphorus
        }
        
        # Apply preprocessing to match main.py's transformations
        soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus = preprocess_input(
            soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus
        )
        
        # Create DataFrame with correct column names (matching main.py)
        input_data = pd.DataFrame({
            'Soil Type': [soil_type],
            'Temparature': [temperature],  # Note: This matches the misspelling in main.py
            'Humidity': [humidity],
            'Moisture': [moisture],
            'Nitrogen': [nitrogen],
            'Potassium': [potassium],
            'Phosphorous': [phosphorus]
        })

        # Encode categorical variables
        input_data_encoded = pd.get_dummies(input_data)
        
        # Handle feature names based on model version
        if hasattr(crop_model, 'feature_names_in_'):
            # For newer scikit-learn versions
            input_data_encoded = input_data_encoded.reindex(columns=crop_model.feature_names_in_, fill_value=0)
        else:
            # For older scikit-learn versions - use the same approach as in main.py
            data = pd.read_csv('data/data.csv')
            data = data.drop(columns=['Fertilizer Name'])
            data_encoded = pd.get_dummies(data, columns=['Soil Type', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'])
            feature_names = data_encoded.drop('Crop Type', axis=1).columns
            input_data_encoded = input_data_encoded.reindex(columns=feature_names, fill_value=0)

        # Make predictions
        crop_prediction = crop_model.predict(input_data_encoded)[0]
        fertilizer_prediction = fertilizer_model.predict(input_data_encoded)[0]

        # Pass all data to template
        return render_template('recommendations.html',
                             crop_prediction=crop_prediction,
                             fertilizer_prediction=fertilizer_prediction,
                             soil_type=original_values['soil_type'],
                             temperature=original_values['temperature'],
                             humidity=original_values['humidity'],
                             moisture=original_values['moisture'],
                             nitrogen=original_values['nitrogen'],
                             potassium=original_values['potassium'],
                             phosphorus=original_values['phosphorus'],
                             algorithm_name=get_algorithm_display_name(current_algorithm))
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # Debug output
        import traceback
        traceback.print_exc()  # Print full error trace for debugging
        return render_template('recommendations.html', error=str(e))

def get_algorithm_display_name(algorithm_code):
    """Convert algorithm code to display name"""
    algorithm_names = {
        'rf': 'Random Forest',
        'gb': 'Gradient Boosting',
        'svm': 'Support Vector Machine',
        'knn': 'K-Nearest Neighbors',
        'mlp': 'Neural Network',
        'lr': 'Logistic Regression'
    }
    return algorithm_names.get(algorithm_code, 'Unknown Algorithm')

@app.route('/model_info')
def model_info():
    """Display information about available models and their performance"""
    models = []
    for algorithm in ['rf', 'gb', 'svm', 'knn', 'mlp', 'lr']:
        crop_path = f'model/crop_model_{algorithm}.pkl'
        fert_path = f'model/fertilizer_model_{algorithm}.pkl'
        if os.path.exists(crop_path) and os.path.exists(fert_path):
            models.append({
                'name': get_algorithm_display_name(algorithm),
                'code': algorithm,
                'is_current': algorithm == current_algorithm
            })
    
    return render_template('model_info.html', models=models, current=current_algorithm)

if __name__ == '__main__':
    app.run(debug=True)