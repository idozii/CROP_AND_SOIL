from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

crop_model = joblib.load('model/crop_model.pkl')
fertilizer_model = joblib.load('model/fertilizer_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    soil_type = request.form['soil_type']
    temperature = request.form['temperature']
    humidity = request.form['humidity']
    moisture = request.form['moisture']
    nitrogen = request.form['nitrogen']
    potassium = request.form['potassium']
    phosphorus = request.form['phosphorus']

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
    input_data_encoded = input_data_encoded.reindex(columns=crop_model.feature_names_in_, fill_value=0)

    crop_prediction = crop_model.predict(input_data_encoded)[0]
    fertilizer_prediction = fertilizer_model.predict(input_data_encoded)[0]

    return render_template('index.html', crop_prediction=crop_prediction, fertilizer_prediction=fertilizer_prediction)

if __name__ == '__main__':
    app.run(debug=True)