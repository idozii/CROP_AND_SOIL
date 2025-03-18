from flask import Flask, request, render_template, redirect, url_for, session, jsonify, flash
import pandas as pd
import joblib
import os
import numpy as np
import functools
import traceback
from main import recommend_crop_ml, recommend_fertilizer_ml, ensemble_predict, analyze_features_impact, get_similar_inputs
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'agriculture_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = password

with app.app_context():
    db.create_all()

crop_model = joblib.load('model/crop_model.pkl')
fertilizer_model = joblib.load('model/fertilizer_model.pkl')
current_algorithm = 'rf' 

def preprocess_input(soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus):    
    try:
        temperature = float(temperature)
        humidity = float(humidity)
        moisture = float(moisture)
        nitrogen = float(nitrogen)
        potassium = float(potassium)
        phosphorus = float(phosphorus)
    except ValueError:
        pass
    
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

def login_required(view_func):
    @functools.wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('home'))
        return view_func(*args, **kwargs)
    return wrapped_view

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is strong"

@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        errors = []
        
        if not name:
            errors.append("Name is required")
        
        if not email:
            errors.append("Email is required")
        elif not validate_email(email):
            errors.append("Invalid email format")
            
        if not password:
            errors.append("Password is required")
        else:
            is_valid, msg = validate_password(password)
            if not is_valid:
                errors.append(msg)
                
        if password != confirm_password:
            errors.append("Passwords don't match")
            
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            errors.append("Email already registered")
            
        if errors:
            for error in errors:
                flash(error, 'error')
            return redirect(url_for('home'))
            
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)

        try:
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('home'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'error')
            print(f"Database error: {str(e)}")
            return redirect(url_for('home'))
    
    return redirect(url_for('home'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Email and password are required', 'error')
            return redirect(url_for('home'))
        
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            flash('Invalid email or password', 'error')
            return redirect(url_for('home'))
        
        session['user_email'] = user.email
        session['user_name'] = user.name
        
        return redirect(url_for('user'))
        
    if 'user_email' in session:
        return redirect(url_for('user'))
        
    return render_template('home.html')

@app.route('/login')
def login():
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('home'))

@app.route('/user')
@login_required
def user():
    algorithm_message = None
    if 'algorithm_message' in session:
        algorithm_message = session['algorithm_message']
        session.pop('algorithm_message', None)
        
    algorithm_name = get_algorithm_display_name(current_algorithm)
    
    return render_template('user.html', 
                          algorithm_message=algorithm_message,
                          current_algorithm=algorithm_name)

@app.route('/real_data')
@login_required
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
@login_required
def recommendations():
    return render_template('recommendations_landing.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/select_algorithm', methods=['GET', 'POST'])
@login_required
def select_algorithm():
    global crop_model, fertilizer_model, current_algorithm
    
    available_algorithms = {
        'rf': 'Random Forest',
        'gb': 'Gradient Boosting',
        'svm': 'Support Vector Machine',
        'knn': 'K-Nearest Neighbors',
        'mlp': 'Neural Network',
        'lr': 'Logistic Regression',
        'ensemble': 'Ensemble (Voting)'
    }
    
    if request.method == 'POST':
        algorithm = request.form.get('algorithm')
        if algorithm in available_algorithms:
            try:
                if algorithm == 'ensemble':
                    current_algorithm = algorithm
                else:
                    crop_model = joblib.load(f'model/crop_model_{algorithm}.pkl')
                    fertilizer_model = joblib.load(f'model/fertilizer_model_{algorithm}.pkl')
                    current_algorithm = algorithm
                session['algorithm_message'] = f"Successfully loaded {available_algorithms[algorithm]} models"
                return redirect(url_for('user'))
                
            except Exception as e:
                return render_template('select_algorithm.html', 
                                      algorithms=available_algorithms,
                                      current=current_algorithm,
                                      error=f"Error loading model: {str(e)}")
    
    return render_template('select_algorithm.html', 
                          algorithms=available_algorithms,
                          current=current_algorithm)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        soil_type = request.form['soil_type']
        temperature = request.form['temperature'] 
        humidity = request.form['humidity']
        moisture = request.form['moisture']
        nitrogen = request.form['nitrogen']
        potassium = request.form['potassium']
        phosphorus = request.form['phosphorus']
        crop_type = request.form.get('crop_type', '')
        
        original_values = {
            'soil_type': soil_type,
            'temperature': temperature,
            'humidity': humidity,
            'moisture': moisture,
            'nitrogen': nitrogen,
            'potassium': potassium,
            'phosphorus': phosphorus,
            'crop_type': crop_type
        }
        
        soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus = preprocess_input(
            soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorus
        )
        
        params = {
            'soil_type': soil_type,
            'temperature': temperature,
            'humidity': humidity,
            'moisture': moisture,
            'nitrogen': nitrogen,
            'potassium': potassium,
            'phosphorus': phosphorus
        }
        
        if current_algorithm == 'ensemble':
            result = ensemble_predict(**params)
            crop_prediction = result['crop_prediction']
            fertilizer_prediction = result['fertilizer_prediction']
            crop_confidence = result['crop_confidence']
            fertilizer_confidence = result['fertilizer_confidence']
            crop_alternatives = result.get('crop_alternatives', [])
            fertilizer_alternatives = result.get('fertilizer_alternatives', [])
            models_used = result.get('models_used', [])
            
            impact_analysis = analyze_features_impact(**params)
            similar_inputs = get_similar_inputs(**params, limit=5)
            
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
                                 crop_type=original_values['crop_type'],
                                 algorithm_name='Ensemble',
                                 crop_confidence=crop_confidence,
                                 fertilizer_confidence=fertilizer_confidence,
                                 crop_alternatives=crop_alternatives,
                                 fertilizer_alternatives=fertilizer_alternatives,
                                 models_used=models_used,
                                 impact_analysis=impact_analysis,
                                 similar_inputs=similar_inputs)
        else:
            if crop_type and crop_type.strip():
                crop_prediction = crop_type.capitalize()
                crop_result = {"prediction": crop_prediction, "confidence": None, "top_features": None}
                fert_result = recommend_fertilizer_ml(model=fertilizer_model, **params)
                fertilizer_prediction = fert_result["prediction"]
            else:
                crop_result = recommend_crop_ml(model=crop_model, **params)
                fert_result = recommend_fertilizer_ml(model=fertilizer_model, **params)
                crop_prediction = crop_result["prediction"]
                fertilizer_prediction = fert_result["prediction"]
                
            impact_analysis = analyze_features_impact(**params)
            similar_inputs = get_similar_inputs(**params, limit=5)

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
                                 crop_type=original_values['crop_type'],
                                 algorithm_name=get_algorithm_display_name(current_algorithm),
                                 crop_confidence=crop_result.get("confidence"),
                                 fertilizer_confidence=fert_result.get("confidence"),
                                 crop_top_features=crop_result.get("top_features"),
                                 fertilizer_top_features=fert_result.get("top_features"),
                                 impact_analysis=impact_analysis,
                                 similar_inputs=similar_inputs)

    except Exception as e:
        print(f"Error in prediction: {str(e)}")  
        traceback.print_exc() 
        return render_template('recommendations.html', error=str(e))

def get_algorithm_display_name(algorithm_code):
    algorithm_names = {
        'rf': 'Random Forest',
        'gb': 'Gradient Boosting',
        'svm': 'Support Vector Machine',
        'knn': 'K-Nearest Neighbors', 
        'mlp': 'Neural Network',
        'lr': 'Logistic Regression',
        'ensemble': 'Ensemble (Voting)'
    }
    return algorithm_names.get(algorithm_code, algorithm_code.upper())

@app.route('/model_info')
@login_required
def model_info():
    algorithms = {
        'rf': 'Random Forest',
        'gb': 'Gradient Boosting',
        'svm': 'Support Vector Machine',
        'knn': 'K-Nearest Neighbors',
        'mlp': 'Neural Network',
        'lr': 'Logistic Regression'
    }
    
    models = []
    for code, name in algorithms.items():
        crop_path = f'model/crop_model_{code}.pkl'
        fert_path = f'model/fertilizer_model_{code}.pkl'
        
        if os.path.exists(crop_path) and os.path.exists(fert_path):
            models.append({
                'code': code,
                'name': name,
                'is_current': code == current_algorithm
            })
    
    return render_template('model_info.html', models=models)

@app.route('/api/feature_impact')
@login_required
def api_feature_impact():
    soil_type = request.args.get('soil_type', 'Sandy')
    temperature = request.args.get('temperature', 'High')
    humidity = request.args.get('humidity', 'Medium')
    moisture = request.args.get('moisture', 'Moderate')
    nitrogen = request.args.get('nitrogen', 'Medium')
    potassium = request.args.get('potassium', 'High')
    phosphorus = request.args.get('phosphorus', 'Low')
    
    params = {
        'soil_type': soil_type,
        'temperature': temperature,
        'humidity': humidity,
        'moisture': moisture,
        'nitrogen': nitrogen,
        'potassium': potassium,
        'phosphorus': phosphorus
    }
    
    impact_analysis = analyze_features_impact(**params)
    return jsonify(impact_analysis)

@app.route('/api/similar_inputs')
@login_required
def api_similar_inputs():
    soil_type = request.args.get('soil_type', 'Sandy')
    temperature = request.args.get('temperature', 'High')
    humidity = request.args.get('humidity', 'Medium')
    moisture = request.args.get('moisture', 'Moderate')
    nitrogen = request.args.get('nitrogen', 'Medium')
    potassium = request.args.get('potassium', 'High')
    phosphorus = request.args.get('phosphorus', 'Low')
    limit = int(request.args.get('limit', 5))
    
    params = {
        'soil_type': soil_type,
        'temperature': temperature,
        'humidity': humidity,
        'moisture': moisture,
        'nitrogen': nitrogen,
        'potassium': potassium,
        'phosphorus': phosphorus,
        'limit': limit
    }
    
    similar_inputs = get_similar_inputs(**params)
    return jsonify(similar_inputs)

if __name__ == '__main__':
    app.run(debug=True)