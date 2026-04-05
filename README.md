# 🌱 AI Crop & Fertilizer Advisor

A simple AI-powered web application that recommends the best crops and fertilizers based on soil and environmental conditions.
[Link](https://crop-and-soil-1.onrender.com/)

## Features

- 🤖 **AI Predictions** - Random Forest machine learning models
- 🌾 **Crop Recommendations** - Suggests optimal crops for your conditions
- 💊 **Fertilizer Recommendations** - Recommends best fertilizers
- 📊 **Confidence Scores** - Shows prediction confidence percentages
- 🎨 **Modern UI** - Clean, responsive dashboard interface
- 🔐 **User Authentication** - Login/signup and secure sessions
- 🧠 **Personalized History** - Saves each user's prediction history
- 📓 **Notebook Inference** - Loads prediction functions from `main.ipynb`

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (from notebook)
```bash
python -c "from notebook_bridge import load_prediction_functions; load_prediction_functions()"
```

### 3. Run App
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## How It Works

1. Enter your soil and environmental data:
   - Soil Type (Sandy, Loamy, Black, Red, Clayey)
   - Temperature (°C)
   - Humidity (%)
   - Soil Moisture (%)
   - Nitrogen, Potassium, Phosphorus levels (mg/kg)

2. Click "Get AI Recommendations"

3. View recommended crop and fertilizer with confidence scores

## Dataset

- **Size**: 8,001 agricultural records
- **Features**: 7 input parameters (soil type, temperature, humidity, moisture, N, P, K)
- **Targets**: Crop types and fertilizer recommendations

## Model Details

- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 decision trees
- **Training**: 80/20 train-test split
- **Features**: Automatic categorical encoding

## Project Structure

```
├── app.py              # Flask web application
├── main.ipynb          # ML notebook (training + prediction functions)
├── notebook_bridge.py  # Safe loader for notebook prediction functions
├── requirements.txt    # Python dependencies
├── vercel.json         # Vercel serverless config
├── render.yaml         # Render deployment config
├── data/
│   └── data.csv       # Agricultural dataset
├── model/             # Trained models (auto-generated)
└── templates/         # HTML templates
     ├── home.html      # Dashboard + input form
     ├── login.html     # Login page
     ├── signup.html    # Signup page
     ├── results.html   # Predictions display
    └── error.html     # Error page
```

## Local vs Cloud Database

- Local development without `DATABASE_URL`: uses SQLite file at `instance/crop_and_soil.db`
- Cloud with `DATABASE_URL`: uses PostgreSQL (recommended for production and Vercel)

## Technologies

- **Backend**: Flask 3.0
- **ML**: scikit-learn, pandas, numpy
- **Frontend**: Bootstrap 5, HTML5
- **Model Storage**: joblib
- **Database**: SQLite (local) / PostgreSQL (cloud)

---
