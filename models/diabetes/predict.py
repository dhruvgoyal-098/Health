import numpy as np
import pickle
import os
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
def load_model_files():
    model_path = os.path.join(MODEL_DIR, 'diabetes.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
model = load_model_files()
DIABETES_FEATURES = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
                     'smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 
                     'smoking_history_never', 'smoking_history_not current']

def predict_diabetes(feature_array):
    feature_array_np = np.array(feature_array, dtype=float).reshape(1, -1)
    prediction = model.predict(feature_array_np)[0]
    try:
        probabilities = model.predict_proba(feature_array_np)[0]
        probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        confidence = float(np.max(probabilities))
    except:
        probability = float(prediction)
        confidence = 0.97
    if prediction == 1:
        risk_level = "High"
        message = "High risk of diabetes. Please consult a doctor for proper diagnosis and management."
    else:
        risk_level = "Low"
        message = "Low risk of diabetes. Maintain a healthy lifestyle with proper diet and exercise."
    
    return {
        'prediction': int(prediction),
        'prediction_label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
        'probability': probability,
        'risk_level': risk_level,
        'message': message,
        'confidence': confidence,
        'error': None
    }
def predict_diabetes_from_dict(feature_dict):
    feature_array = [float(feature_dict.get(name, 0)) for name in DIABETES_FEATURES]
    return predict_diabetes(feature_array)