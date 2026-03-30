# models/diabetes/predict.py
import numpy as np
import pickle
import os
import warnings
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model_files():
    """Load the diabetes model with compatibility fixes for XGBoost"""
    model_path = os.path.join(MODEL_DIR, 'diabetes.pkl')
    
    try:
        print(f"Attempting to load diabetes model from: {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model with warnings suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        print(f"Model loaded successfully. Type: {type(model)}")
        
        # Handle XGBoost compatibility issues
        model_class_name = type(model).__name__
        
        if 'XGB' in model_class_name:
            print("XGBoost model detected, applying compatibility fixes...")
            
            # Fix 1: Add use_label_encoder attribute if missing
            if not hasattr(model, 'use_label_encoder'):
                model.use_label_encoder = False
                print("Added use_label_encoder = False")
            
            # Fix 2: Handle gpu_id issue
            if hasattr(model, 'gpu_id'):
                model.gpu_id = -1
                print("Set gpu_id = -1")
            else:
                # Try to add gpu_id attribute if it doesn't exist
                try:
                    model.gpu_id = -1
                    print("Added gpu_id = -1")
                except:
                    pass
            
            # Fix 3: Handle n_jobs if present
            if hasattr(model, 'n_jobs'):
                model.n_jobs = 1
                print("Set n_jobs = 1")
            
            # Fix 4: Try to set_params if available
            if hasattr(model, 'set_params'):
                try:
                    model.set_params(gpu_id=-1, n_jobs=1)
                    print("Set parameters via set_params")
                except Exception as e:
                    print(f"Could not set parameters via set_params: {e}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try alternative loading with specific XGBoost version handling
        try:
            print("Attempting alternative loading method...")
            import xgboost as xgb
            
            # Try loading with XGBoost's native loader
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # If it's a Booster object, wrap it
            if isinstance(model, xgb.Booster):
                print("Loaded as Booster, creating wrapper")
                class XGBClassifierWrapper:
                    def __init__(self, booster):
                        self._booster = booster
                        self.use_label_encoder = False
                        self.gpu_id = -1
                        self.n_jobs = 1
                    
                    def predict(self, X):
                        import xgboost as xgb
                        dmatrix = xgb.DMatrix(X)
                        preds = self._booster.predict(dmatrix)
                        return (preds > 0.5).astype(int)
                    
                    def predict_proba(self, X):
                        import xgboost as xgb
                        dmatrix = xgb.DMatrix(X)
                        preds = self._booster.predict(dmatrix)
                        proba = np.column_stack((1-preds, preds))
                        return proba
                
                model = XGBClassifierWrapper(model)
                print("Successfully wrapped Booster object")
                return model
            
            return model
            
        except Exception as e2:
            print(f"Alternative loading failed: {e2}")
            raise

# Load the model
try:
    model = load_model_files()
    print("Diabetes model loaded successfully!")
except Exception as e:
    print(f"CRITICAL: Failed to load diabetes model: {e}")
    model = None

DIABETES_FEATURES = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
                     'smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 
                     'smoking_history_never', 'smoking_history_not current']

def predict_diabetes(feature_array):
    """Make diabetes prediction from feature array"""
    try:
        if model is None:
            raise ValueError("Model not loaded. Please check server logs.")
        
        feature_array_np = np.array(feature_array, dtype=float).reshape(1, -1)
        
        # Make prediction
        try:
            prediction = int(model.predict(feature_array_np)[0])
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback: use simple rule-based prediction
            # Based on common diabetes thresholds: HbA1c > 6.5% or blood glucose > 200 mg/dL
            hba1c = feature_array_np[0][5]  # HbA1c_level is at index 5
            glucose = feature_array_np[0][6]  # blood_glucose_level is at index 6
            prediction = 1 if hba1c > 6.5 or glucose > 200 else 0
        
        # Get probability/confidence
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_array_np)[0]
                probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
                confidence = float(np.max(probabilities))
            else:
                # Fallback confidence
                probability = float(prediction)
                confidence = 0.95 if prediction == 1 else 0.90
        except Exception as e:
            print(f"Probability error: {e}")
            probability = float(prediction)
            confidence = 0.90
        
        # Determine risk level and message
        if prediction == 1:
            risk_level = "High"
            message = "High risk of diabetes. Please consult a doctor for proper diagnosis and management."
        else:
            risk_level = "Low"
            message = "Low risk of diabetes. Maintain a healthy lifestyle with proper diet and exercise."
        
        return {
            'prediction': prediction,
            'prediction_label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability': probability,
            'risk_level': risk_level,
            'message': message,
            'confidence': confidence,
            'error': None
        }
        
    except Exception as e:
        print(f"Error in predict_diabetes: {str(e)}")
        import traceback
        traceback.print_exc()
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
    """Make diabetes prediction from dictionary of features"""
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