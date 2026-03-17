import numpy as np
import pickle
import os
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model_files():
    model_path = os.path.join(MODEL_DIR, 'kidney.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    encoders_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    with open(encoders_path, 'rb') as file:
        label_encoders = pickle.load(file)
    return model, scaler, label_encoders

model, scaler, label_encoders = load_model_files()

print(f"Kidney Model expects: {model.n_features_in_} features")
print(f"Scaler expects: {scaler.n_features_in_} features")

# Print target mapping for debugging
target_encoder = label_encoders.get('Target')
target_mapping = {}
if target_encoder:
    target_classes = target_encoder.classes_
    print(f"Target classes: {target_classes}")
    target_mapping = {i: str(cls) for i, cls in enumerate(target_classes)}
    print(f"Target mapping: {target_mapping}")

# Indices of numerical features (0-27)
NUMERICAL_INDICES = list(range(28))

# Categorical column names in order
CATEGORICAL_NAMES = [
    'Red blood cells in urine',
    'Pus cells in urine',
    'Pus cell clumps in urine',
    'Bacteria in urine',
    'Hypertension (yes/no)',
    'Diabetes mellitus (yes/no)',
    'Coronary artery disease (yes/no)',
    'Appetite (good/poor)',
    'Pedal edema (yes/no)',
    'Anemia (yes/no)',
    'Family history of chronic kidney disease',
    'Smoking status',
    'Physical activity level',
    'Urinary sediment microscopy results'
]

def safe_encode(value, encoder):
    """Safely encode a value without any iteration errors"""
    try:
        # Convert to string and clean
        str_value = str(value).strip().lower()
        
        # Get classes as list of strings
        classes = []
        for c in encoder.classes_:
            classes.append(str(c).strip().lower())
        
        # Manual search without using 'in' operator
        for i, cls in enumerate(classes):
            if cls == str_value:
                return i
        
        # If not found, return 0
        return 0
    except:
        return 0

def predict_kidney_disease(feature_array):
    if model is None or scaler is None or label_encoders is None:
        return {
            'error': 'Model files not loaded properly',
            'prediction': None,
            'probability': None,
            'risk_level': None,
            'message': None,
            'confidence': None
        }
    
    expected_features = model.n_features_in_
    
    if len(feature_array) != expected_features:
        return {
            'error': f'Expected {expected_features} features, got {len(feature_array)}',
            'prediction': None,
            'probability': None,
            'risk_level': None,
            'message': None,
            'confidence': None
        }
    
    try:
        # Separate numerical and categorical features
        numerical_values = []
        categorical_values = []
        
        for i, val in enumerate(feature_array):
            if i < 28:  # Numerical
                try:
                    numerical_values.append(float(val))
                except:
                    numerical_values.append(0.0)
            else:  # Categorical
                categorical_values.append(val)
        
        # Create DataFrame for numerical values with proper column names
        if hasattr(scaler, 'feature_names_in_'):
            numerical_df = pd.DataFrame(
                [numerical_values],
                columns=scaler.feature_names_in_
            )
            numerical_scaled = scaler.transform(numerical_df)
        else:
            # Fallback to numpy array if scaler doesn't have feature names
            numerical_array = np.array(numerical_values).reshape(1, -1)
            numerical_scaled = scaler.transform(numerical_array)
        
        # Encode categorical values safely
        categorical_encoded = []
        
        for i, raw_val in enumerate(categorical_values):
            if i < len(CATEGORICAL_NAMES):
                col_name = CATEGORICAL_NAMES[i]
                # Use .get() to safely access encoder
                encoder = label_encoders.get(col_name)
                if encoder is not None:
                    encoded = safe_encode(raw_val, encoder)
                else:
                    encoded = 0
            else:
                encoded = 0
            categorical_encoded.append(encoded)
        
        categorical_array = np.array(categorical_encoded).reshape(1, -1)
        
        # Combine features
        final_features = np.hstack((numerical_scaled, categorical_array))
        
        # Make prediction
        prediction = model.predict(final_features)[0]
        
        # Get probabilities
        try:
            probabilities = model.predict_proba(final_features)[0]
            if len(probabilities) > 1:
                probability = float(probabilities[1])
                confidence = float(np.max(probabilities))
            else:
                probability = float(probabilities[0])
                confidence = probability
        except:
            probability = float(prediction)
            confidence = 0.80
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'error': f'Prediction error: {str(e)}',
            'prediction': None,
            'probability': None,
            'risk_level': None,
            'message': None,
            'confidence': None
        }
    
    # Get human-readable prediction label
    target_encoder = label_encoders.get('Target')
    if target_encoder is not None:
        target_labels = target_encoder.classes_
        prediction_label = str(target_labels[prediction])
    else:
        prediction_label = f"Class {prediction}"
    
    # FIX: Handle numeric labels properly
    # Check if prediction_label is actually a number string like '0', '1', etc.
    if prediction_label.isdigit():
        pred_value = int(prediction_label)
        # Map based on common patterns
        if pred_value == 0:
            prediction_label="Prediction : NO"
            risk_level = "None"
            message = "No kidney disease detected. Maintain a healthy lifestyle."
        elif pred_value == 1:
            prediction_label="Prediction : YES"
            risk_level = "Low"
            message = "Low risk of kidney disease. Maintain regular checkups and healthy lifestyle."
        elif pred_value == 2:
            prediction_label="Prediction : YES"
            risk_level = "Moderate"
            message = "Moderate risk of kidney disease. Consult a doctor for further evaluation."
        elif pred_value >= 3:
            risk_level = "High"
            message = f"High risk of kidney disease detected (Level {pred_value}). Please consult a nephrologist immediately."
        else:
            risk_level = "Unknown"
            message = f"Prediction: {prediction_label}. Please consult a doctor for interpretation."
    else:
        # Original string-based detection for text labels
        prediction_upper = prediction_label.upper()
        
        if 'HIGH' in prediction_upper:
            risk_level = "High"
            message = f"High risk of kidney disease detected: {prediction_label}. Please consult a nephrologist immediately."
        elif 'LOW' in prediction_upper:
            risk_level = "Low"
            message = f"Low risk of kidney disease: {prediction_label}. Maintain regular checkups and healthy lifestyle."
        elif 'NO' in prediction_upper or 'NONE' in prediction_upper:
            risk_level = "None"
            message = f"No kidney disease detected: {prediction_label}. Maintain a healthy lifestyle."
        elif 'MODERATE' in prediction_upper:
            risk_level = "Moderate"
            message = f"Moderate risk of kidney disease: {prediction_label}. Consult a doctor for further evaluation."
        else:
            risk_level = "Unknown"
            message = f"Prediction: {prediction_label}. Please consult a doctor for interpretation."
    
    return {
        'prediction': int(prediction),
        'prediction_label': prediction_label,
        'probability': probability,
        'risk_level': risk_level,
        'message': message,
        'confidence': confidence,
        'error': None
    }

def predict_kidney_from_dict(feature_dict):
    # Feature names in correct order
    feature_names = [
        'Age of the patient', 'Blood pressure (mm/Hg)', 'Specific gravity of urine',
        'Albumin in urine', 'Sugar in urine', 'Red blood cells in urine',
        'Pus cells in urine', 'Pus cell clumps in urine', 'Bacteria in urine',
        'Random blood glucose level (mg/dl)', 'Blood urea (mg/dl)',
        'Serum creatinine (mg/dl)', 'Sodium level (mEq/L)', 'Potassium level (mEq/L)',
        'Hemoglobin level (gms)', 'Packed cell volume (%)',
        'White blood cell count (cells/cumm)', 'Red blood cell count (millions/cumm)',
        'Hypertension (yes/no)', 'Diabetes mellitus (yes/no)',
        'Coronary artery disease (yes/no)', 'Appetite (good/poor)',
        'Pedal edema (yes/no)', 'Anemia (yes/no)',
        'Estimated Glomerular Filtration Rate (eGFR)', 'Urine protein-to-creatinine ratio',
        'Urine output (ml/day)', 'Serum albumin level', 'Cholesterol level',
        'Parathyroid hormone (PTH) level', 'Serum calcium level', 'Serum phosphate level',
        'Family history of chronic kidney disease', 'Smoking status',
        'Body Mass Index (BMI)', 'Physical activity level',
        'Duration of diabetes mellitus (years)', 'Duration of hypertension (years)',
        'Cystatin C level', 'Urinary sediment microscopy results',
        'C-reactive protein (CRP) level', 'Interleukin-6 (IL-6) level'
    ]
    
    feature_array = []
    for name in feature_names:
        value = feature_dict.get(name, '0')
        feature_array.append(value)
    
    return predict_kidney_disease(feature_array)