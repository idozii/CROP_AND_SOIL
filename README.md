# 🌱 AI Crop & Fertilizer Advisor

A simple AI-powered web application that recommends the best crops and fertilizers based on soil and environmental conditions.
[Link](https://crop-and-soil-1.onrender.com/)

## Features

- 🤖 **AI Predictions** - Random Forest machine learning models
- 🌾 **Crop Recommendations** - Suggests optimal crops for your conditions
- 💊 **Fertilizer Recommendations** - Recommends best fertilizers
- 📊 **Confidence Scores** - Shows prediction confidence percentages
- 🎨 **Modern UI** - Clean, responsive Bootstrap 5 interface

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python main.py
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
├── main.py             # ML model training & prediction
├── requirements.txt    # Python dependencies
├── data/
│   └── data.csv       # Agricultural dataset
├── model/             # Trained models (auto-generated)
└── templates/         # HTML templates
    ├── home.html      # Input form
    ├── results.html   # Predictions display
    └── error.html     # Error page
```

## Deploy to Vercel

1. Install Vercel CLI: `npm i -g vercel`
2. Run: `vercel --prod`

## Technologies

- **Backend**: Flask 3.0
- **ML**: scikit-learn, pandas, numpy
- **Frontend**: Bootstrap 5, HTML5
- **Model Storage**: joblib

---
