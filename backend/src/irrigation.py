# backend/src/irrigation.py

def recommend_timing(moisture_pct, rainfall_mm):
    """
    Recommends watering timing based on soil moisture and rain.
    """
    
    if rainfall_mm > 10:
        return [
            "Heavy rain detected.",
            "No irrigation needed for at least 3-4 days."
        ]
        
    if moisture_pct > 60:
        return [
            "Soil is very moist.",
            "No irrigation needed. Check again in 3-4 days."
        ]
    elif moisture_pct > 40:
        return [
            "Soil has adequate moisture.",
            "Check again in 2-3 days."
        ]
    elif moisture_pct > 25:
        return [
            "Soil is drying.",
            "Plan to water in the next 1-2 days."
        ]
    else:
        return [
            "Soil is dry!",
            "Immediate watering is recommended."
        ]