# backend/src/soil.py
import pickle
import numpy as np
import pandas as pd
import os

# Load the trained model and the feature list
model = None
model_features = []
try:
    with open('backend/models/fertilizer_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('backend/models/fertilizer_features.pkl', 'rb') as f:
        model_features = pickle.load(f)
    print("Fertilizer model and features loaded successfully.")
except FileNotFoundError:
    print("ERROR: fertilizer_model.pkl or fertilizer_features.pkl not found. Run model_trainer.py")

def recommend(soil_data):
    """
    Uses the trained ML model to recommend fertilizer.
    """
    if not model:
        return ["Fertilizer Model not loaded."]
        
    try:
        # 1. Create a DataFrame with all 0s, matching the training features
        input_df = pd.DataFrame(columns=model_features)
        input_df.loc[0] = 0
        
        print("--- Making Soil Prediction ---")
        print(f"UI Data: {soil_data}")

        # 2. Map data from the UI (which uses 'N', 'P', 'K')
        # to the model's columns ('Nitrogen', 'Potassium', 'Phosphorous')
        # We use .get(key, 0) to provide a default value if the key is missing
        input_df['Nitrogen'] = float(soil_data.get("N", 0))
        input_df['Potassium'] = float(soil_data.get("K", 0))
        input_df['Phosphorous'] = float(soil_data.get("P", 0))
        
        # 3. Add placeholders for data not on the UI.
        print("WARNING: Using placeholder data for Temp, Humidity, Moisture, Soil Type, and Crop Type.")
        input_df['Temparature'] = 25.0 # Placeholder
        input_df['Humidity'] = 60.0    # Placeholder
        input_df['Moisture'] = 40.0    # Placeholder
        
        # 4. Set default dummy variables (placeholders)
        if 'Soil Type_Loamy' in input_df.columns:
            input_df['Soil Type_Loamy'] = 1
        if 'Crop Type_Maize' in input_df.columns:
            input_df['Crop Type_Maize'] = 1

        # 5. Make prediction
        prediction = model.predict(input_df)
        
        return [f"Recommended fertilizer: {prediction[0]}"]

    except Exception as e:
        print(f"Fertilizer prediction error: {e}")
        return ["Error predicting fertilizer."]