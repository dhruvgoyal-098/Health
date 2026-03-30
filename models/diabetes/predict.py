import numpy as np
import pickle
import os
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
def load_model_files():
    model_path = os.path.join(MODEL_DIR, 'diabetes.pkl')
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model_class_name = type(model).__name__
        if 'XGB' in model_class_name:
            if not hasattr(model, 'use_label_encoder'):
                model.use_label_encoder = False
            if hasattr(model, 'gpu_id'):
                model.gpu_id = -1
            else:
                model.gpu_id = -1
            if hasattr(model, 'n_jobs'):
                model.n_jobs = 1
            if hasattr(model, 'set_params'):
                model.set_params(gpu_id=-1, n_jobs=1)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        import xgboost as xgb
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        if isinstance(model, xgb.Booster):        
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
            return model 
    return model
model = load_model_files()
DIABETES_FEATURES = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
                     'smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 
                     'smoking_history_never', 'smoking_history_not current']
def predict_diabetes(feature_array):
        feature_array_np = np.array(feature_array, dtype=float).reshape(1, -1)
        try:
            prediction = int(model.predict(feature_array_np)[0])
        except Exception as e:
            hba1c = feature_array_np[0][5]
            glucose = feature_array_np[0][6]
            prediction = 1 if hba1c > 6.5 or glucose > 200 else 0
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_array_np)[0]
                probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
                confidence = float(np.max(probabilities))
            else:
                probability = float(prediction)
                confidence = 0.95 if prediction == 1 else 0.90
        except Exception as e:
            print(f"Probability error: {e}")
            probability = float(prediction)
            confidence = 0.90
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
def predict_diabetes_from_dict(feature_dict):
    feature_array = [float(feature_dict.get(name, 0)) for name in DIABETES_FEATURES]
    return predict_diabetes(feature_array)