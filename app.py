#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
from flasgger import Swagger
import os

app = Flask(__name__)
Swagger(app)

# Load the trained coefficients (beta_manual)
with open("manual_regression_model.pkl", "rb") as f:
    classifier = pickle.load(f)  # this is a numpy array of shape (7, 1)

@app.route('/')
def welcome():
    return "Welcome to the Poverty Prediction API!"

@app.route('/predict', methods=["GET"])
def predict_poverty():
    """Predict Poverty Status
    ---
    parameters:
      - name: pop_chng
        in: query
        type: number
        required: true
      - name: n_empld
        in: query
        type: number
        required: true
      - name: tax_rate
        in: query
        type: number
        required: true
      - name: pt_phone
        in: query
        type: number
        required: true
      - name: pt_rural
        in: query
        type: number
        required: true
      - name: age
        in: query
        type: number
        required: true
    responses:
        200:
            description: Predicted Poverty Percentage
    """
    try:
        pop_chng = float(request.args.get("pop_chng"))
        n_empld = float(request.args.get("n_empld"))
        tax_rate = float(request.args.get("tax_rate"))
        pt_phone = float(request.args.get("pt_phone"))
        pt_rural = float(request.args.get("pt_rural"))
        age = float(request.args.get("age"))

        X = np.array([1, pop_chng, n_empld, tax_rate, pt_phone, pt_rural, age]).reshape(1, -1)
        prediction = float(X @ classifier)

        return f"Predicted Poverty Percentage: {round(prediction, 2)}%"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/predict_file', methods=["POST"])
def predict_poverty_file():
    """Predict Poverty Status from File
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: Predicted Poverty Percentages
    """
    try:
        df_test = pd.read_csv(request.files.get("file"))
        X_test = np.c_[np.ones(df_test.shape[0]), df_test.values]
        predictions = X_test @ classifier
        return str(predictions.flatten().round(2).tolist())
    except Exception as e:
        return f"Error: {str(e)}"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # default to 10000 if PORT not set
    app.run(host="0.0.0.0", port=port)







