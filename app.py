"""
Optimized Flask app for Vercel deployment
Lightweight, fast, and production-ready
"""
from flask import Flask, request, render_template, redirect, url_for, session, jsonify, flash
import os
import logging
import traceback
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import re

# Import optimized ML functions
from main_optimized import predict_crop, predict_fertilizer, get_data_sample

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'agriculture_secret_key_2024')

# Database configuration
database_path = os.path.join(BASE_DIR, 'instance', 'users.db')
os.makedirs(os.path.dirname(database_path), exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{database_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

class User(db.Model):
    """User model for authentication"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = password

# Create tables
with app.app_context():
    db.create_all()

# Utility functions
def login_required(f):
    """Decorator for login required routes"""
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password: str) -> tuple:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    
    return True, "Password is strong"

def preprocess_form_input(form_data: dict) -> dict:
    """Preprocess form input for ML models"""
    try:
        # Map form values to model expected values
        processed = {}
        
        # Handle soil type
        processed['soil_type'] = form_data.get('soil_type', 'Sandy')
        
        # Handle categorical temperature
        temp_val = form_data.get('temperature', '25')
        try:
            temp_num = float(temp_val)
            if temp_num < 20:
                processed['temperature'] = 'Low'
            elif temp_num > 30:
                processed['temperature'] = 'High'
            else:
                processed['temperature'] = 'Medium'
        except:
            processed['temperature'] = temp_val
        
        # Handle categorical humidity
        humidity_val = form_data.get('humidity', '50')
        try:
            humidity_num = float(humidity_val)
            if humidity_num < 40:
                processed['humidity'] = 'Low'
            elif humidity_num > 70:
                processed['humidity'] = 'High'
            else:
                processed['humidity'] = 'Medium'
        except:
            processed['humidity'] = humidity_val
        
        # Handle categorical moisture
        moisture_val = form_data.get('moisture', '40')
        try:
            moisture_num = float(moisture_val)
            if moisture_num < 30:
                processed['moisture'] = 'Low'
            elif moisture_num > 60:
                processed['moisture'] = 'High'
            else:
                processed['moisture'] = 'Moderate'
        except:
            processed['moisture'] = moisture_val
        
        # Handle NPK values
        for nutrient in ['nitrogen', 'potassium', 'phosphorus']:
            val = form_data.get(nutrient, '20')
            try:
                num_val = float(val)
                if num_val < 15:
                    processed[nutrient] = 'Low'
                elif num_val > 30:
                    processed[nutrient] = 'High'
                else:
                    processed[nutrient] = 'Medium'
            except:
                processed[nutrient] = val
        
        return processed
        
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        return form_data

# Routes
@app.route('/', methods=['GET', 'POST'])
def home():
    """Home page with login/register"""
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
        
        return redirect(url_for('dashboard'))
        
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
        
    return render_template('home.html')

@app.route('/register', methods=['POST'])
def register():
    """User registration"""
    try:
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
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('home'))
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Registration error: {e}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html', user_name=session.get('user_name'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Crop and fertilizer prediction"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Get form data
        form_data = {
            'soil_type': request.form.get('soil_type'),
            'temperature': request.form.get('temperature'),
            'humidity': request.form.get('humidity'),
            'moisture': request.form.get('moisture'),
            'nitrogen': request.form.get('nitrogen'),
            'potassium': request.form.get('potassium'),
            'phosphorus': request.form.get('phosphorus')
        }
        
        # Preprocess input
        processed_data = preprocess_form_input(form_data)
        
        # Make predictions
        crop_result = predict_crop(
            processed_data['soil_type'],
            processed_data['temperature'], 
            processed_data['humidity'],
            processed_data['moisture'],
            processed_data['nitrogen'],
            processed_data['potassium'],
            processed_data['phosphorus']
        )
        
        fertilizer_result = predict_fertilizer(
            processed_data['soil_type'],
            processed_data['temperature'],
            processed_data['humidity'], 
            processed_data['moisture'],
            processed_data['nitrogen'],
            processed_data['potassium'],
            processed_data['phosphorus']
        )
        
        return render_template('results.html',
                             crop_result=crop_result,
                             fertilizer_result=fertilizer_result,
                             input_data=form_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        traceback.print_exc()
        flash(f'Prediction error: {str(e)}', 'error')
        return render_template('predict.html')

@app.route('/data')
@login_required  
def view_data():
    """View sample data"""
    try:
        data_info = get_data_sample(limit=100)
        return render_template('data.html', data_info=data_info)
    except Exception as e:
        logger.error(f"Data view error: {e}")
        flash(f'Error loading data: {str(e)}', 'error')
        return render_template('data.html', data_info={'error': str(e)})

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        crop_result = predict_crop(
            data.get('soil_type'),
            data.get('temperature'),
            data.get('humidity'),
            data.get('moisture'),
            data.get('nitrogen'),
            data.get('potassium'),
            data.get('phosphorus')
        )
        
        fertilizer_result = predict_fertilizer(
            data.get('soil_type'),
            data.get('temperature'),
            data.get('humidity'),
            data.get('moisture'),
            data.get('nitrogen'),
            data.get('potassium'),
            data.get('phosphorus')
        )
        
        return jsonify({
            'crop': crop_result,
            'fertilizer': fertilizer_result,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return render_template('error.html', error="Internal server error"), 500

# For local development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)