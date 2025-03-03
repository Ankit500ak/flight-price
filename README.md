# Flight Price Prediction Web Application 🛫💰

## Overview

This is an advanced Flight Price Prediction web application that leverages machine learning to predict flight prices based on temporal features. The application provides an intuitive web interface for making predictions and visualizing model performance.

![Project Banner](https://img.shields.io/badge/Status-Active-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.8+-blue)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-orange)
![Web Framework](https://img.shields.io/badge/Web-Flask-green)

## 🌟 Key Features

- **Machine Learning Prediction**: Uses RandomForestRegressor for flight price prediction
- **Interactive Web Interface**: Modern, responsive design with smooth animations
- **Model Performance Visualization**:
  - Feature Importance Plot
  - Prediction Scatter Plot
  - Learning Curves
- **Detailed Performance Metrics**
- **Responsive Design**: Works on desktop and mobile devices

## 🛠 Technology Stack

- **Backend**: 
  - Python
  - Flask
  - Scikit-learn
- **Frontend**:
  - HTML5
  - CSS3
  - Vanilla JavaScript
- **Data Processing**:
  - Pandas
  - NumPy
- **Visualization**:
  - Matplotlib
  - Seaborn

## 📦 Prerequisites

- Python 3.8+
- pip (Python Package Manager)
- Virtual Environment (recommended)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/flight-price-predictor.git
cd flight-price-predictor
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python train_model.py
```

### 5. Run the Application
```bash
python app.py
```

🌐 Open your browser and navigate to `http://127.0.0.1:5000/`

## 🔍 How It Works

### Data Preparation
- Generates synthetic flight data with features:
  - Year
  - Month
  - Day
  - Hour
  - Minute

### Model Training
- Uses RandomForestRegressor
- Performs feature scaling
- Generates performance metrics and visualizations

### Prediction Process
1. User inputs flight temporal details
2. Data is preprocessed and scaled
3. Model predicts the flight price
4. Results are displayed with confidence metrics

## 📊 Model Performance Metrics

The application provides:
- Mean Squared Error (MSE)
- R² Score
- Feature Importance Breakdown

## 🎨 User Interface

### Prediction Section
- Intuitive input form for flight details
- Real-time price prediction
- Error handling and informative messages

### Performance Metrics Section
- Toggleable performance insights
- Visual plots for model understanding
- Detailed feature importance analysis

## 🔒 Security & Best Practices

- Input validation
- Error handling
- Secure model loading
- Logging for debugging

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙌 Acknowledgments

- [Scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Matplotlib](https://matplotlib.org/)

---

**Disclaimer**: This is a demonstration project using synthetic data. Real-world flight price prediction requires extensive historical data and complex feature engineering.

## 📞 Contact

Your Name - youremail@example.com

Project Link: [https://github.com/yourusername/flight-price-predictor](https://github.com/yourusername/flight-price-predictor)
