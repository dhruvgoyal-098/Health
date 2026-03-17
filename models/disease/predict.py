import numpy as np
import joblib
import json
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model_files():
    model_path = os.path.join(MODEL_DIR, 'disease_model.pkl')
    encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    medicine_path = os.path.join(MODEL_DIR, 'medicine_map.json')
    
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    with open(medicine_path, 'r') as f:
        medicine_map = json.load(f)
    return model, label_encoder, medicine_map

model, label_encoder, medicine_map = load_model_files()

SYMPTOM_COLUMNS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
    'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue',
    'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
    'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
    'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain',
    'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
    'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
    'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
    'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
    'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
    'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine',
    'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
    'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
    'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze',
    'Urinating_a_lot', 'Heartburn'
]

def predict_disease(symptom_array):
    if model is None or label_encoder is None or medicine_map is None:
        return {
            'error': 'Model files not loaded properly',
            'disease': None,
            'medicine': None,
            'confidence': None
        }
    
    expected_features = model.n_features_in_
    
    if not isinstance(symptom_array, (list, np.ndarray)):
        return {
            'error': f'Expected list or array, got {type(symptom_array)}',
            'disease': None,
            'medicine': None,
            'confidence': None
        }
    
    if len(symptom_array) != expected_features:
        return {
            'error': f'Expected {expected_features} symptoms, got {len(symptom_array)}',
            'disease': None,
            'medicine': None,
            'confidence': None
        }
    
    symptom_array_np = np.array(symptom_array, dtype=float).reshape(1, -1)
    
    try:
        prediction = model.predict(symptom_array_np)
        prediction_proba = model.predict_proba(symptom_array_np)
        disease_name = label_encoder.inverse_transform(prediction)[0]
        recommended_medicine = medicine_map.get(disease_name, "Consult a doctor")
        confidence = float(np.max(prediction_proba))
    except Exception as e:
        return {
            'error': f'Prediction error: {str(e)}',
            'disease': None,
            'medicine': None,
            'confidence': None
        }
    
    return {
        'disease': disease_name,
        'medicine': recommended_medicine,
        'confidence': confidence,
        'error': None
    }

def predict_disease_from_dict(symptom_dict):
    symptom_array = [float(symptom_dict.get(symptom, 0)) for symptom in SYMPTOM_COLUMNS]
    return predict_disease(symptom_array)

def get_symptom_descriptions():
    return {
        'symptoms': SYMPTOM_COLUMNS,
        'count': len(SYMPTOM_COLUMNS)
    }