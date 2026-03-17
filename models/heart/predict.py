import numpy as np
import joblib
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model_files():
    model_path = os.path.join(MODEL_DIR, 'heart_txt.pkl')
    model = joblib.load(model_path)
    return model

heart_model = load_model_files()

# These are for reference only - not used in prediction
FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

def predict_heart_disease(feature_array):
    if heart_model is None:
        return {
            'error': 'Model not loaded properly',
            'prediction': None,
            'probability': None,
            'risk_level': None,
            'message': None,
            'confidence': None
        }
    
    # Ensure we have a list of numbers
    if not isinstance(feature_array, (list, np.ndarray)):
        return {
            'error': f'Expected list or array, got {type(feature_array)}',
            'prediction': None,
            'probability': None,
            'risk_level': None,
            'message': None,
            'confidence': None
        }
    
    expected_features = heart_model.n_features_in_
    
    if len(feature_array) != expected_features:
        return {
            'error': f'Expected {expected_features} features, got {len(feature_array)}',
            'prediction': None,
            'probability': None,
            'risk_level': None,
            'message': None,
            'confidence': None
        }
    
    # Convert to numpy array with float dtype - this is the key part
    # We create a pure numpy array with no column names
    feature_array_np = np.array(feature_array, dtype=float).reshape(1, -1)
    
    try:
        # Make prediction - this should work if the model was saved correctly
        prediction = heart_model.predict(feature_array_np)[0]
        probability = heart_model.predict_proba(feature_array_np)[0]
        disease_probability = float(probability[1])
        confidence = float(np.max(probability))
    except Exception as e:
        # If it fails, try to diagnose
        error_msg = str(e)
        print(f"Prediction error: {error_msg}")
        
        # Try alternative approach if the error mentions columns
        if 'columns' in error_msg.lower() or 'dataframe' in error_msg.lower():
            try:
                # Some models need the exact feature names
                import pandas as pd
                feature_df = pd.DataFrame([feature_array], columns=FEATURE_NAMES)
                prediction = heart_model.predict(feature_df)[0]
                probability = heart_model.predict_proba(feature_df)[0]
                disease_probability = float(probability[1])
                confidence = float(np.max(probability))
            except Exception as e2:
                return {
                    'error': f'Prediction error: {str(e2)}',
                    'prediction': None,
                    'probability': None,
                    'risk_level': None,
                    'message': None,
                    'confidence': None
                }
        else:
            return {
                'error': f'Prediction error: {error_msg}',
                'prediction': None,
                'probability': None,
                'risk_level': None,
                'message': None,
                'confidence': None
            }
    
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
    # Create a list of values in the correct order
    feature_array = []
    for name in FEATURE_NAMES:
        value = feature_dict.get(name, 0)
        try:
            feature_array.append(float(value))
        except (ValueError, TypeError):
            feature_array.append(0.0)
    
    return predict_heart_disease(feature_array)