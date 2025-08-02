from flask import Flask, request, jsonify
import pickle
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS  # Import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os


# Add these imports BEFORE loading pickled models
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

load_dotenv()  


api_key = os.getenv("API_KEY")
app = Flask(__name__)
CORS(app)

# Set your Gemini API key
genai.configure(api_key=api_key)
# Load your model at startup
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Mlmodels/depression_model.pkl', 'rb') as f:
    depression_model = joblib.load(f)['model']

with open('Mlmodels/anxiety_model.pkl', 'rb') as f:
    anxiety_model = joblib.load(f)['model']      

print("Model expects these features:", model)
@app.route('/', methods=['GET'])
def home():
    return 'Hello from API Server'

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        data = request.json
        prompt = data.get("prompt")  

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Generate response from Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        return jsonify({"response": response.text})  

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/predict', methods=['POST'])
def predict():
    # List of required fields and their expected data types
    required_fields = {      
        'Age': int,
        'Academic_Pressure': int,  # Scale (presumably 0-4/5)
        'Cgpa': float,
        'Study_Satisfaction': int,  # Scale (presumably 0-5)  
        'Dietary_Habits': int,      # 0=Unhealthy, 1=Moderate, 2=Healthy
        'Suicidal_Thoughts': int,    # 0=No, 1=Yes
        'WrkStdy_Hours': int,
        'Financial_Stress': int,     # Scale (presumably 0-5)
     
    }

    # Get and validate JSON data
    data = request.get_json(force=True)
    
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400

    # Validate data types
    type_errors = []
    for field, expected_type in required_fields.items():
        value = data.get(field)
        if not isinstance(value, expected_type):
            type_errors.append(f'{field} should be {expected_type.__name__}')
    
    if type_errors:
        return jsonify({'error': 'Type errors', 'details': type_errors}), 400

    try:
        # Convert input data to DataFrame in correct feature order
        input_df = pd.DataFrame([data])[required_fields.keys()]

        # Convert to numpy array and reshape for prediction
        features = input_df.values.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist() if hasattr(model, 'predict_proba') else None

        response = {
            'success': 'true',
            'prediction': int(prediction),
            'probability': probability
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e),'success': 'false'}), 500


@app.route('/depression', methods=['POST'])
def depression_predict():
    # List of required fields and their expected data types
    required_fields = ['anxiety', 'benzodiazepine_use', 'tobacco_use', 'unemployed', 'rural', 'low_education', 'average_income', 'is_school_dropout', 'was_child_married', 'has_chronic_disease', 'exposed_to_domestic_violence', 'aware_of_mental_health', 'has_been_a_crime_victim', 'can_access_healthcare', 'has_mental_health_facility_access', 'has_internet', 'orphan', 'individual_married', 'individual_employed', 'family_liability', 'has_bank_loan', 'gender_encoded']


    # Get and validate JSON data
    data = request.get_json(force=True)
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400    

    try:
        # Convert input data to DataFrame in correct feature order
        input_df = pd.DataFrame([data])[required_fields]

        # Convert to numpy array and reshape for prediction
        features = input_df.values.reshape(1, -1)
        
        # Make prediction
        prediction = depression_model.predict(features)[0]
        probability = depression_model.predict_proba(features)[0].tolist() if hasattr(depression_model, 'predict_proba') else None

        response = {
            'success': 'true',
            'prediction': int(prediction),
            'probability': probability
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e), 'success': 'false'}), 500

@app.route('/anxiety', methods=['POST'])
def anxiety():
    try:
        features = ['ptsd', 'benzodiazepine_use', 'tobacco_use', 'unemployed', 'rural', 'low_education', 'average_income', 'is_school_dropout', 'was_child_married', 'has_chronic_disease', 'exposed_to_domestic_violence', 'aware_of_mental_health', 'has_been_a_crime_victim', 'can_access_healthcare', 'has_mental_health_facility_access', 'has_internet', 'orphan', 'individual_married', 'individual_employed', 'family_liability', 'has_bank_loan', 'gender_encoded']
        # Get and validate JSON data
        data = request.get_json(force=True)
        # Check for missing fields
        missing_fields = [field for field in features if field not in data]     
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        # Convert input data to DataFrame in correct feature order
        input_df = pd.DataFrame([data])[features]
        # Convert to numpy array and reshape for prediction
        features = input_df.values.reshape(1, -1)
        # Make prediction
        prediction = anxiety_model.predict(features)[0]
        probability = anxiety_model.predict_proba(features)[0].tolist() if hasattr(anxiety_model, 'predict_proba') else None
        response = {
            'success': 'true',
            'prediction': int(prediction),
            'probability': probability
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e), 'success': 'false'}), 500




if __name__ == '__main__':
    app.run(debug=True)
