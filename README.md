# 🌱 AI Crop & Fertilizer Recommendation System

A machine learning-powered web application that provides intelligent crop and fertilizer recommendations based on soil conditions and environmental factors.

## 🎯 Key Features

### 🤖 AI-Powered Recommendations
- **Crop Prediction**: Suggests optimal crops based on soil and environmental conditions
- **Fertilizer Recommendation**: Recommends appropriate fertilizers for specific soil compositions
- **Confidence Scoring**: Provides confidence percentages for each recommendation
- **Real-time Analysis**: Instant predictions with optimized machine learning models

### 🔐 User Management
- **Secure Authentication**: User registration and login system
- **Password Validation**: Strong password requirements with real-time feedback
- **Session Management**: Secure user sessions with Flask-SQLAlchemy
- **User Dashboard**: Personalized interface for each user

### 📊 Data Visualization
- **Interactive Forms**: User-friendly soil analysis input forms
- **Progress Indicators**: Visual confidence bars for predictions
- **Dataset Explorer**: Browse and search through the training dataset
- **Responsive Design**: Mobile-friendly Bootstrap 5 interface

## 📈 Machine Learning Models

### Algorithm: Random Forest Classifier
- **Type**: Ensemble learning method using decision trees
- **Configuration**: 50 estimators for optimal performance-accuracy balance
- **Training Strategy**: Separate models for crop and fertilizer predictions
- **Optimization**: Lightweight design suitable for serverless deployment

### Model Performance
- **Training Dataset**: 20% sample (1,600 records) for faster training
- **Accuracy**: High prediction accuracy with comprehensive validation
- **Memory Efficiency**: Compressed models (~5MB) optimized for Vercel deployment
- **Preprocessing**: LabelEncoder for categorical features, ensuring robust handling of soil types and nutrient levels

### Model Features
- **Input Parameters**: 7 key agricultural factors
  - Soil Type (Sandy, Loamy, Black, Red, Clayey)
  - Temperature (°C)
  - Humidity (%)
  - Soil Moisture (%)
  - Nitrogen content (mg/kg)
  - Potassium content (mg/kg)
  - Phosphorus content (mg/kg)

- **Output Predictions**:
  - **Crop Types**: Maize, Sugarcane, Cotton, Tobacco, Paddy, Barley, Wheat, Millets, and more
  - **Fertilizer Types**: Urea, DAP, 14-35-14, 28-28, 17-17-17, 20-20, and specialized blends

## 📊 Dataset Information

### Data Source
- **Size**: 8,001 agricultural records
- **Origin**: Comprehensive agricultural database with real-world farming data
- **Coverage**: Multiple soil types, climate conditions, and geographic regions

### Data Structure
```
Features (Input):
├── Environmental Factors
│   ├── Temperature (°C): 10-50°C range
│   ├── Humidity (%): 20-100% relative humidity
│   └── Moisture (%): 10-80% soil moisture content
├── Soil Characteristics
│   ├── Soil Type: 5 major soil classifications
│   ├── Nitrogen (N): 0-100 mg/kg
│   ├── Potassium (K): 0-100 mg/kg
│   └── Phosphorus (P): 0-100 mg/kg
└── Target Variables (Output)
    ├── Crop Type: 22+ different crop varieties
    └── Fertilizer Name: 15+ fertilizer formulations
```

### Data Quality
- **Completeness**: No missing values, all records contain complete information
- **Diversity**: Covers various agricultural scenarios and conditions
- **Preprocessing**: Categorical encoding and numerical normalization
- **Validation**: Cross-validated for model training and testing

## � Technical Architecture

### Backend Stack
- **Framework**: Flask 3.0 (Python web framework)
- **Database**: SQLite with Flask-SQLAlchemy ORM
- **ML Library**: Scikit-learn for machine learning models
- **Data Processing**: Pandas and NumPy for data manipulation

### Frontend Stack
- **UI Framework**: Bootstrap 5 for responsive design
- **Icons**: Bootstrap Icons for visual elements
- **JavaScript**: Vanilla JS for form validation and interactions
- **Styling**: Custom CSS with modern design principles

### Deployment Optimization
- **Platform**: Optimized for Vercel serverless deployment
- **Dependencies**: Minimal package requirements (7 essential libraries)
- **Performance**: Model caching with singleton pattern
- **Memory**: Efficient memory usage under 1GB limit

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Train Models
```bash
python main.py
```

### Run Locally
```bash
python app.py
```

### Deploy to Vercel
```bash
vercel --prod
```

## 📦 Dependencies
- Flask==3.0.0 (Web framework)
- Flask-SQLAlchemy==3.1.1 (Database ORM)
- pandas==2.1.4 (Data manipulation)
- numpy==1.24.3 (Numerical computing)
- scikit-learn==1.3.2 (Machine learning)
- joblib==1.3.2 (Model serialization)
- Werkzeug==3.0.1 (WSGI utilities)

## 🎯 Use Cases

### For Farmers
- Get crop recommendations based on current soil conditions
- Optimize fertilizer usage for better yield
- Make data-driven agricultural decisions

### For Agricultural Consultants
- Provide scientific recommendations to clients
- Analyze soil conditions efficiently
- Support sustainable farming practices

### For Researchers
- Explore agricultural datasets
- Understand crop-soil relationships
- Validate farming hypotheses with data

---

**Built with ❤️ for sustainable agriculture and smart farming**