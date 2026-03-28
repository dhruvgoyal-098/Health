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
target_encoder = label_encoders.get('Target')
target_mapping = {}
if target_encoder:
    target_classes = target_encoder.classes_
    print(f"Target classes: {target_classes}")
    target_mapping = {i: str(cls) for i, cls in enumerate(target_classes)}
    print(f"Target mapping: {target_mapping}")
NUMERICAL_INDICES = list(range(28))
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
    try:
        str_value = str(value).strip().lower()
        classes = []
        for c in encoder.classes_:
            classes.append(str(c).strip().lower())
        for i, cls in enumerate(classes):
            if cls == str_value:
                return i
        return 0
    except:
        return 0
def predict_kidney_disease(feature_array):
    numerical_values = []
    categorical_values = []
    for i, val in enumerate(feature_array):
        if i < 28:
            try:
                numerical_values.append(float(val))
            except:
                numerical_values.append(0.0)
        else: 
            categorical_values.append(val)
    if hasattr(scaler, 'feature_names_in_'):
        numerical_df = pd.DataFrame(
            [numerical_values],
            columns=scaler.feature_names_in_
        )
        numerical_scaled = scaler.transform(numerical_df)
    else:
        numerical_array = np.array(numerical_values).reshape(1, -1)
        numerical_scaled = scaler.transform(numerical_array)
    categorical_encoded = []
    for i, raw_val in enumerate(categorical_values):
        if i < len(CATEGORICAL_NAMES):
            col_name = CATEGORICAL_NAMES[i]
            encoder = label_encoders.get(col_name)
            if encoder is not None:
                encoded = safe_encode(raw_val, encoder)
            else:
                encoded = 0
        else:
            encoded = 0
        categorical_encoded.append(encoded)
    categorical_array = np.array(categorical_encoded).reshape(1, -1)
    final_features = np.hstack((numerical_scaled, categorical_array))
    prediction = model.predict(final_features)[0]
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
    target_encoder = label_encoders.get('Target')
    if target_encoder is not None:
        target_labels = target_encoder.classes_
        prediction_label = str(target_labels[prediction])
    else:
        prediction_label = f"Class {prediction}"
    if prediction_label.isdigit():
        pred_value = int(prediction_label)
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