from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Global variables for model and scaler
model = None
scaler = None
performance_metrics = {}

# Function to load model and scaler
def load_model_and_scaler():
    global model, scaler, performance_metrics
    try:
        # Check if files exist
        if not os.path.exists('model.pkl'):
            app.logger.error("model.pkl does not exist. Please train the model first.")
            return False
        
        if not os.path.exists('scaler.pkl'):
            app.logger.error("scaler.pkl does not exist. Please train the model first.")
            return False
        
        # Load model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load performance metrics
        if os.path.exists('performance_metrics.json'):
            with open('performance_metrics.json', 'r') as f:
                performance_metrics = json.load(f)
        else:
            app.logger.warning("performance_metrics.json not found.")
        
        app.logger.info("Model and scaler loaded successfully")
        return True
    except Exception as e:
        app.logger.error(f"Error loading model or scaler: {str(e)}")
        model = None
        scaler = None
        performance_metrics = {}
        return False

# Load model and scaler on startup
load_model_and_scaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            # Try to reload
            if not load_model_and_scaler():
                return jsonify({
                    'error': 'Model not trained. Please train the model first.',
                    'details': 'Run train_model.py to train the model before making predictions.'
                }), 500
        
        data = request.get_json()
        app.logger.info(f"Received data: {data}")
        
        # Extract features in correct order
        features = [
            float(data.get('year', 2024)),    # Year
            float(data.get('month', 1)),      # Month
            float(data.get('day', 1)),        # Day
            float(data.get('hour', 0)),       # Hour
            float(data.get('minute', 0))      # Minute
        ]
        
        # Scale the features
        input_data = scaler.transform(np.array(features).reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'input_features': features
        })
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'An error occurred during prediction',
            'details': str(e)
        }), 400

@app.route('/model_performance')
def model_performance():
    """Return model performance metrics and visualizations"""
    try:
        # Check if performance metrics exist
        if not performance_metrics:
            return jsonify({
                'error': 'No performance metrics available',
                'details': 'Please train the model first by running train_model.py'
            }), 404

        # Read pre-generated visualizations
        def encode_image(filename):
            try:
                if not os.path.exists(filename):
                    app.logger.warning(f"Image file {filename} not found")
                    return None
                with open(filename, 'rb') as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            except Exception as e:
                app.logger.error(f"Error reading {filename}: {str(e)}")
                return None

        return jsonify({
            'performance_metrics': performance_metrics,
            'feature_importance_plot': encode_image('feature_importance.png'),
            'prediction_scatter_plot': encode_image('prediction_scatter.png'),
            'learning_curves_plot': encode_image('learning_curves.png')
        })
    except Exception as e:
        app.logger.error(f"Model performance error: {str(e)}")
        return jsonify({
            'error': 'An error occurred retrieving model performance',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
