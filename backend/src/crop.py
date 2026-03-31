# backend/src/crop.py
import pickle
import numpy as np
import os

# Load the trained model
model_path = 'backend/models/crop_model.pkl'
model = None

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Crop model loaded successfully.")
else:
    print(f"ERROR: Crop model not found at {model_path}. Run model_trainer.py")

def recommend_crop(data):
    """
    Uses the trained ML model to predict a crop.
    """
    if not model:
        return "Crop Model not loaded."
    
    try:
        # 1. Prepare data in the correct order (must match training)
        # 1. Prepare data in the correct order (must match training)
        features = [
            data.get("N", 90),
            data.get("P", 40),
            data.get("K", 40),
            data.get("temp", 25),
            data.get("humidity", 60),
            data.get("ph", 6.5), # <-- FIX: 'pH' is now 'ph'
            data.get("rainfall", 100)
        ]
        
        # 2. Convert to 2D numpy array for sklearn
        final_features = [np.array(features)]
        
        # 3. Make prediction
        prediction = model.predict(final_features)
        
        print(f"Crop prediction: {prediction[0]}")
        return prediction[0] # Return the first (and only) prediction

    except Exception as e:
        print(f"Crop prediction error: {e}")
        return "Error predicting crop."