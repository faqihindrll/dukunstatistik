from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import joblib
import pandas as pd

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
            'XGBoost': joblib.load(os.path.join(MODEL_PATH, 'XGBoost_model.pkl')),
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
def predict_arimaxx():
    # Mengambil data JSON dari permintaan
    data = request.get_json()
    input_values = data['values']

    # Mengubah query menjadi array numpy
    exog = np.array(input_values).reshape(1, -1)

    # Melakukan prediksi
    forecast = modelkemiskinan.predict(n_periods=1, exogenous=exog)

    # Menangani berbagai kemungkinan output
    if isinstance(forecast, pd.Series):
        result = forecast.iloc[0]
    elif isinstance(forecast, np.ndarray):
        result = forecast[0]
    elif isinstance(forecast, (float, int)):
        result = forecast
    else:
        raise ValueError(f"Unexpected forecast type: {type(forecast)}")

    return jsonify(result)


@app.route('/')
def home():
    return render_template('project/public/index.html')

if __name__ == '__main__':
    app.run(debug=True)