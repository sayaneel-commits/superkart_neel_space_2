import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
superkart_api = Flask("SuperKart")
MODEL_PATH = "superkart_production_pipeline.pkl"

# Check if the model file exists, then load it
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model_pipeline = pickle.load(file)
else:
    raise FileNotFoundError(f"CRITICAL ERROR: Production pipeline file missing at '{MODEL_PATH}'")

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System - Forecasting API is Live."

# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        data = request.get_json()
        
        # Extract features and map them to match our exact Phase 2 training features
        sample = {
            'Product_Weight': float(data['Product_Weight']),
            'Product_Sugar_Content': str(data['Product_Sugar_Content']),
            'Product_Allocated_Area': float(data['Product_Allocated_Area']),
            'Product_MRP': float(data['Product_MRP']),
            'Store_Size': str(data['Store_Size']),
            'Store_Type': str(data['Store_Type']),
            'Product_ID_Type': str(data['Product_Id_char']),      # Remapped to training name
            'Store_Age': int(data['Store_Age_Years']),            # Remapped to training name
            'Product_Perishability': str(data['Product_Type_Category']) # Remapped to training name
        }

        # Convert to DataFrame matching model signature (Store_Location_City_Type explicitly excluded)
        input_data = pd.DataFrame([sample])

        # Run pipeline inference pass
        raw_prediction = model_pipeline.predict(input_data)[0]
        
        # Apply physical floor boundary check ($0.00 base revenue)
        sanitized_prediction = round(max(0.0, float(raw_prediction)), 2)

        return jsonify({'Sales': sanitized_prediction}), 200
        
    except Exception as error:
        return jsonify({'status': 'error', 'message': str(error)}), 400

# Define an endpoint to predict sales for a batch of products via uploaded CSV file
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        file = request.files['file']
        input_data = pd.read_csv(file)

        # Essential Column Name Mapping Realignment to match training schema
        rename_map = {
            'Product_Id_char': 'Product_ID_Type',
            'Store_Age_Years': 'Store_Age',
            'Product_Type_Category': 'Product_Perishability'
        }
        input_data = input_data.rename(columns=rename_map)

        # Explicitly drop the perfectly collinear column if present to match transformer signature
        if 'Store_Location_City_Type' in input_data.columns:
            input_data = input_data.drop(columns=['Store_Location_City_Type'])

        # Filter out metadata identifying columns if present to prevent pipeline error drops
        for redundant_col in ['Product_Id', 'Store_Id', 'Product_Type', 'Store_Establishment_Year']:
            if redundant_col in input_data.columns:
                input_data = input_data.drop(columns=[redundant_col])

        # Make predictions for the batch data rows
        raw_predictions = model_pipeline.predict(input_data)
        
        # Enforce mathematical lower boundaries and compile clean key-value mapping indices
        output_dict = {
            str(i): round(max(0.0, float(pred)), 2) 
            for i, pred in enumerate(raw_predictions)
        }

        return jsonify(output_dict), 200
        
    except Exception as error:
        return jsonify({'status': 'error', 'message': str(error)}), 400

# Run the Flask app on internal port 5000 inside multi-container networks
if __name__ == '__main__':
    superkart_api.run(debug=True)
