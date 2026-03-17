from flask import Flask, render_template, request, jsonify
import sys
import os

import models.heart.predict as hp
import models.lung.predict as lp
import models.diabetes.predict as dp
import models.kidney.predict as kp
import models.disease.predict as dsp

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        data = request.json
        
        if 'features' in data:
            features = data['features']
            result = hp.predict_heart_disease(features)
        elif 'patient_data' in data:
            patient_dict = data['patient_data']
            result = hp.predict_heart_from_dict(patient_dict)
        else:
            return jsonify({'error': 'No features provided'}), 400
        
        if result.get('error'):
            return jsonify({'error': result['error']}), 400
        
        return jsonify({
            'success': True,
            'prediction': 'Heart Disease Detected' if result['prediction'] == 1 else 'No Heart Disease',
            'probability': f"{result['probability']:.2%}",
            'risk_level': result['risk_level'],
            'message': result['message'],
            'confidence': f"{result['confidence']:.2%}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/lung')
def lung():
    return render_template('lung.html')

@app.route('/predict/lung', methods=['POST'])
def predict_lung():
    try:
        data = request.json
        
        if 'features' in data:
            features = data['features']
            result = lp.predict_lung_cancer(features)
        elif 'patient_data' in data:
            patient_dict = data['patient_data']
            result = lp.predict_lung_from_dict(patient_dict)
        else:
            return jsonify({'error': 'No features provided'}), 400
        
        if result.get('error'):
            return jsonify({'error': result['error']}), 400
        
        return jsonify({
            'success': True,
            'prediction': result['prediction_label'],
            'risk_level': result['risk_level'],
            'message': result['message'],
            'confidence': f"{result['confidence']:.2%}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json
        
        if 'features' in data:
            features = data['features']
            result = dp.predict_diabetes(features)
        elif 'patient_data' in data:
            patient_dict = data['patient_data']
            result = dp.predict_diabetes_from_dict(patient_dict)
        else:
            return jsonify({'error': 'No features provided'}), 400
        
        if result.get('error'):
            return jsonify({'error': result['error']}), 400
        
        return jsonify({
            'success': True,
            'prediction': result['prediction_label'],
            'risk_level': result['risk_level'],
            'message': result['message'],
            'confidence': f"{result['confidence']:.2%}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

@app.route('/predict/kidney', methods=['POST'])
def predict_kidney():
    try:
        data = request.json
        
        if 'features' in data:
            features = data['features']
            result = kp.predict_kidney_disease(features)
        elif 'patient_data' in data:
            patient_dict = data['patient_data']
            result = kp.predict_kidney_from_dict(patient_dict)
        else:
            return jsonify({'error': 'No features provided'}), 400
        
        if result.get('error'):
            return jsonify({'error': result['error']}), 400
        
        return jsonify({
            'success': True,
            'prediction': result['prediction_label'],
            'risk_level': result['risk_level'],
            'message': result['message'],
            'confidence': f"{result['confidence']:.2%}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    try:
        data = request.json
        
        if 'symptoms' in data:
            symptoms = data['symptoms']
            result = dsp.predict_disease(symptoms)
        elif 'symptom_dict' in data:
            symptom_dict = data['symptom_dict']
            result = dsp.predict_disease_from_dict(symptom_dict)
        else:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        if result.get('error'):
            return jsonify({'error': result['error']}), 400
        
        return jsonify({
            'success': True,
            'disease': result['disease'],
            'medicine': result['medicine'],
            'confidence': f"{result['confidence']:.2%}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    try:
        symptoms = dsp.get_symptom_descriptions()
        return jsonify({
            'success': True,
            'symptoms': symptoms['symptoms'],
            'count': symptoms['count']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/hospitals')
def hospitals():
    return render_template('hospitals.html')

@app.route('/news')
def news():
    return render_template('news.html')

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)