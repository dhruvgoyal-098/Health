import numpy as np
import joblib
import os
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(MODEL_DIR, 'heart_txt.pkl')
heart_model = joblib.load(model_path)
FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
def predict_heart_disease(feature_array):
    feature_array_np = np.array(feature_array, dtype=float).reshape(1, -1)
    try:
        prediction = heart_model.predict(feature_array_np)[0]
        probability = heart_model.predict_proba(feature_array_np)[0]
        disease_probability = float(probability[1])
        confidence = float(np.max(probability))
    except Exception as e:
        error_msg = str(e)
        print(f"Prediction error: {error_msg}")
        import pandas as pd
        feature_df = pd.DataFrame([feature_array], columns=FEATURE_NAMES)
        prediction = heart_model.predict(feature_df)[0]
        probability = heart_model.predict_proba(feature_df)[0]
        disease_probability = float(probability[1])
        confidence = float(np.max(probability))
    if disease_probability < 0.3:
        risk_level = "Low"
        message = "Low risk of heart disease. Maintain healthy lifestyle."
    elif disease_probability < 0.6:
        risk_level = "Moderate"
        message = "Moderate risk of heart disease. Consider lifestyle changes and consult a doctor."
    else:
        risk_level = "High"
        message = "High risk of heart disease. Please consult a cardiologist immediately."
    return {
        'prediction': int(prediction),
        'probability': disease_probability,
        'risk_level': risk_level,
        'message': message,
        'confidence': confidence,
        'error': None
    }
def predict_heart_from_dict(feature_dict):
    feature_array = []
    for name in FEATURE_NAMES:
        value = feature_dict.get(name, 0)
        try:
            feature_array.append(float(value))
        except (ValueError, TypeError):
            feature_array.append(0.0)
    return predict_heart_disease(feature_array)