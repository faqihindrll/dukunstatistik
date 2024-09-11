from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

app = Flask(__name__)

# Tentukan path absolut ke model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')

@app.route('/predictkemiskinan', methods=['POST'])
def predictedK():
    # Mengambil data JSON dari permintaan
    data = request.get_json()
    input_values = data.get('values')

    if input_values is None:
        return jsonify({'error': 'Missing "values" in request data'}), 400

    try:
        # Memuat model, scaler, dan PCA
        scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
        pca = joblib.load(os.path.join(MODEL_PATH, 'pca.pkl'))
        pcs_above_one = joblib.load(os.path.join(MODEL_PATH, 'pcs_above_one.pkl'))

        best_models = {
            'Random Forest': joblib.load(os.path.join(MODEL_PATH, 'Random Forest_model.pkl')),
            'XGBoost': joblib.load(os.path.join(MODEL_PATH, 'XGBoost_model.pkl'))
        }

        # Transformasi input
        column_names = ['Biaya kuliah', 'Biaya listrik', 'Biaya rumah sakit', 'Medical check up','Paket Internet','Biaya sekolah','Harga bensin','Jual rumah']  # Ganti dengan nama kolom yang sesuai
        input_df = pd.DataFrame([input_values], columns=column_names)

        # Transformasi input
        new_input_scaled = scaler.transform(input_df)
        new_input_pca = pca.transform(new_input_scaled)[:, pcs_above_one]

        # Melakukan prediksi dengan semua model
        forecast_results = {}
        for name, model in best_models.items():
            forecast_value = model.predict(new_input_pca)
            forecast_results[name] = forecast_value.tolist()

        return jsonify(forecast_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictpengangguran', methods=['POST'])
def predictedP():
    # Mengambil data JSON dari permintaan
    data = request.get_json()
    input_values = data.get('values')
    
    if input_values is None:
        return jsonify({'error': 'Missing "values" in request data'}), 400

    try:
        num_features = len(input_values)  # Adjust this if you have a fixed number of features
        input_df = pd.DataFrame([num_features], columns=['Lowongan Kerja'])  # Replace with actual feature names
        
        best_models = {
            'XGBoost': joblib.load(os.path.join(MODEL_PATH, 'XGBoost_Pengangguran.pkl')),
            'Random Forest': joblib.load(os.path.join(MODEL_PATH, 'Random Forest_Pengangguran.pkl'))
        }
        forecast_results = {}
        for name, model in best_models.items():
            forecast_value = model.predict(input_df)
            forecast_results[name] = forecast_value.tolist()
            
        return jsonify(forecast_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('project/public/index.html')

if __name__ == '__main__':
    app.run(debug=True)