from flask import Flask, request, render_template
from main import predict_crop, predict_fertilizer
import os

app = Flask(__name__)

# Ensure models are trained on startup
@app.before_request
def ensure_models_exist():
    """Train models if they don't exist (first request only)"""
    if not hasattr(app, 'models_checked'):
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        crop_model_path = os.path.join(model_dir, 'crop_model.pkl')
        
        if not os.path.exists(crop_model_path):
            print("🔧 Models not found, training now...")
            from main import train_models
            train_models()
            print("✅ Models ready!")
        
        app.models_checked = True

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions"""
    try:
        data = {
            'soil_type': request.form.get('soil_type'),
            'temperature': request.form.get('temperature'),
            'humidity': request.form.get('humidity'),
            'moisture': request.form.get('moisture'),
            'nitrogen': request.form.get('nitrogen'),
            'potassium': request.form.get('potassium'),
            'phosphorus': request.form.get('phosphorus')
        }
        
        crop_result = predict_crop(
            data['soil_type'],
            data['temperature'],
            data['humidity'],
            data['moisture'],
            data['nitrogen'],
            data['potassium'],
            data['phosphorus']
        )
        
        fertilizer_result = predict_fertilizer(
            data['soil_type'],
            data['temperature'],
            data['humidity'],
            data['moisture'],
            data['nitrogen'],
            data['potassium'],
            data['phosphorus']
        )
        
        return render_template('results.html',
                             crop=crop_result,
                             fertilizer=fertilizer_result,
                             input_data=data)
    except Exception as e:
        return render_template('error.html', error=str(e))

app_handler = app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
