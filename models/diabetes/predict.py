# models/diabetes/predict.py
import numpy as np
import pickle
import os
import warnings

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model_files():
    try:
        model_path = os.path.join(MODEL_DIR, 'diabetes.pkl')
        print(f"Loading diabetes model from: {model_path}")
        
        # Suppress warnings when loading the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        # If it's an XGBoost model, we need to handle it specially
        if 'XGBClassifier' in str(type(model)):
            # For XGBoost models, we might need to set attributes
            try:
                # Try to set use_label_encoder if it's missing (for compatibility)
                if not hasattr(model, 'use_label_encoder'):
                    model.use_label_encoder = False
            except:
                pass
        
        return model
    except Exception as e:
        print(f"Error loading diabetes model: {str(e)}")
        # Try alternative loading method
        try:
            import xgboost as xgb
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            # For newer XGBoost versions, we need to handle this
            if hasattr(model, '_Booster'):
                # This is an XGBoost model
                pass
            return model
        except Exception as e2:
            print(f"Alternative loading also failed: {str(e2)}")
            raise

model = load_model_files()

DIABETES_FEATURES = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
                     'smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 
                     'smoking_history_never', 'smoking_history_not current']

def predict_diabetes(feature_array):
    try:
        feature_array_np = np.array(feature_array, dtype=float).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(feature_array_np)[0]
        
        # Try to get probabilities
        try:
            # For XGBoost models, we need to handle predict_proba
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_array_np)[0]
                probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
                confidence = float(np.max(probabilities))
            else:
                # Fallback for models without predict_proba
                probability = float(prediction)
                confidence = 0.95
        except Exception as e:
            print(f"Error getting probabilities: {e}")
            probability = float(prediction)
            confidence = 0.95
        
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
    except Exception as e:
        print(f"Error in predict_diabetes: {str(e)}")
        return {
            'prediction': 0,
            'prediction_label': 'Error',
            'probability': 0,
            'risk_level': 'Error',
            'message': f"Prediction error: {str(e)}",
            'confidence': 0,
            'error': str(e)
        }

def predict_diabetes_from_dict(feature_dict):
    try:
        feature_array = [float(feature_dict.get(name, 0)) for name in DIABETES_FEATURES]
        return predict_diabetes(feature_array)
    except Exception as e:
        print(f"Error in predict_diabetes_from_dict: {str(e)}")
        return {
            'prediction': 0,
            'prediction_label': 'Error',
            'probability': 0,
            'risk_level': 'Error',
            'message': f"Error processing data: {str(e)}",
            'confidence': 0,
            'error': str(e)
        }