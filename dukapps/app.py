from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import joblib
import pandas as pd


app = Flask(__name__)

# Tentukan path absolut ke model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model_arimax.joblib')

# Debugging: print path model
print(f"Model path: {MODEL_PATH}")

# Memuat model
try:
    modelkemiskinan = joblib.load(MODEL_PATH)
    modelpengangguran = joblib.load(MODEL_PATH)
except FileNotFoundError as e:
    print(f"File not found: {e}")
    raise

@app.route('/predictpengagguran', methods=['POST'])
def predict_arimax():
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

@app.route('/predictkemiskinan', methods=['POST'])
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