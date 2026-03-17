import numpy as np
import joblib
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model_files():
    model_path = os.path.join(MODEL_DIR, 'lung_txt.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_files()

LUNG_FEATURES = ['SMOKING', 'YELLOW_FINGERS', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

ACTUAL_FEATURES = LUNG_FEATURES[:model.n_features_in_]

def predict_lung_cancer(feature_array):
    if model is None or scaler is None:
        return {
            'error': 'Model files not loaded properly',
            'prediction': None,
            'probability': None,
            'risk_level': None,
            'message': None,
            'confidence': None
        }
    
    expected_features = model.n_features_in_
    
    if not isinstance(feature_array, (list, np.ndarray)):
        return {
            'error': f'Expected list or array, got {type(feature_array)}',
            'prediction': None,
            'probability': None,
            'risk_level': None,
            'message': None,
            'confidence': None
        }
    
    if len(feature_array) >= expected_features:
        selected_features = feature_array[:expected_features]
    else:
        return {
            'error': f'Expected at least {expected_features} features, got {len(feature_array)}',
            'prediction': None,
            'probability': None,
            'risk_level': None,
            'message': None,
            'confidence': None
        }
    
    feature_array_np = np.array(selected_features, dtype=float).reshape(1, -1)
    feature_array_scaled = scaler.transform(feature_array_np)
    
    try:
        prediction = model.predict(feature_array_scaled)[0]
        try:
            probabilities = model.predict_proba(feature_array_scaled)[0]
            probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
            confidence = float(np.max(probabilities))
        except:
            probability = float(prediction)
            confidence = 0.85
    except Exception as e:
        return {
            'error': f'Prediction error: {str(e)}',
            'prediction': None,
            'probability': None,
            'risk_level': None,
            'message': None,
            'confidence': None
        }
    
    if prediction == 1:
        risk_level = "High"
        message = "High risk of lung cancer. Please consult a pulmonologist immediately for further evaluation."
    else:
        risk_level = "Low"
        message = "Low risk of lung cancer. Maintain a healthy lifestyle and avoid smoking."
    
    return {
        'prediction': int(prediction),
        'prediction_label': 'YES' if prediction == 1 else 'NO',
        'probability': probability,
        'risk_level': risk_level,
        'message': message,
        'confidence': confidence,
        'error': None
    }

def predict_lung_from_dict(feature_dict):
    feature_array = [float(feature_dict.get(name, 0)) for name in ACTUAL_FEATURES]
    return predict_lung_cancer(feature_array)