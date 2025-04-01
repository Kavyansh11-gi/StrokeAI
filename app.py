import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import os
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'RFmodel.pkl')


with open(model_path, 'rb') as model_file:
    model = joblib.load(model_file)

template_dir = os.path.abspath('templates')
app = Flask(__name__, template_folder=template_dir, static_folder='static')


gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
hypertension_map = {'No': 0, 'Yes': 1}
heart_disease_map = {'No': 0, 'Yes': 1}
ever_married_map = {'No': 1, 'Yes': 0}
work_type_map = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
residence_type_map = {'Urban': 1, 'Rural': 0}
smoking_status_map = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    input_data = []

    for key, value in data.items():
        if key == 'bmi' and value is None:
            value = 28.89
        if key in ['age', 'avg_glucose_level', 'bmi']:
            input_data.append(float(value))
        elif key in ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
            if key == 'gender':
                input_data.append(gender_map.get(value, 0))
            elif key == 'hypertension':
                input_data.append(hypertension_map.get(value, 0))
            elif key == 'heart_disease':
                input_data.append(heart_disease_map.get(value, 0))
            elif key == 'ever_married':
                input_data.append(ever_married_map.get(value, 0))
            elif key == 'work_type':
                input_data.append(work_type_map.get(value, 0))
            elif key == 'Residence_type':
                input_data.append(residence_type_map.get(value, 0))
            elif key == 'smoking_status':
                input_data.append(smoking_status_map.get(value, 0))
    
    input_data = np.array(input_data).reshape(1, -1)

    print(input_data)

    print('Input Data:', input_data)
    
    # Make prediction
    output = model.predict(input_data)
    print('Prediction:', output)

    result = int(output[0])

    return jsonify({'result': result})

@app.route('/causes')
def causes():
    if request.method == 'GET':
        return render_template('causes.html')

@app.route('/prevention')
def prevention():
    if request.method == 'GET':
     return render_template('prevention.html')

@app.route('/stayInformed')
def stay_informed():
    if request.method == 'GET':
     return render_template('stayInformed.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predicted_text = ''
    if request.method == 'POST':
        gender = request.form.get('gender')
        age = float(request.form.get('age', '0'))
        hypertension = request.form.get('hypertension')
        heart_disease = request.form.get('heart_disease')
        ever_married = request.form.get('ever_married')
        work_type = request.form.get('work_type')
        residence_type = request.form.get('residence_type')
        avg_glucose_level = float(request.form.get('avg_glucose_level', '0.0'))
        bmi = float(request.form.get('bmi', '0.0'))
        smoking_status = request.form.get('smoking_status')

        gender_encoded = gender_map.get(gender, 0)
        hypertension_encoded = hypertension_map.get(hypertension, 0)
        heart_disease_encoded = heart_disease_map.get(heart_disease, 0)
        ever_married_encoded = ever_married_map.get(ever_married, 0)
        work_type_encoded = work_type_map.get(work_type, 0)
        residence_type_encoded = residence_type_map.get(residence_type, 0)
        smoking_status_encoded = smoking_status_map.get(smoking_status, 0)

        input_data = [gender_encoded, age, hypertension_encoded, heart_disease_encoded, ever_married_encoded,
                      work_type_encoded, residence_type_encoded, avg_glucose_level, bmi, smoking_status_encoded]
        input_data = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_data)
        prediction = int(prediction[0])

        print('Form Data:', request.form)
        print('Input Data:', input_data)
        print('Prediction:', prediction)

        if( prediction == 0):
            predicted_text = 'The prediction indicates a low risk of stroke. To maintain this, focus on a balanced diet rich in fruits, vegetables, and whole grains. Regular physical activity and avoiding smoking can also significantly reduce the risk. Monitoring blood pressure and cholesterol levels is essential, along with managing stress effectively. Regular check-ups with healthcare providers can help ensure early detection and prevention.'
        else:
            predicted_text = 'Our prediction indicates a potential risk of stroke. Its imperative to promptly seek medical advice for a comprehensive assessment and appropriate management. To reduce this risk, prioritize a healthy lifestyle with a balanced diet, regular physical activity, and avoiding smoking. Managing underlying conditions like hypertension and diabetes is also crucial. Consistent monitoring and adherence to medical guidance can significantly mitigate the risk of stroke and promote overall health.'

    return render_template('predict.html', predicted_text=predicted_text)

if __name__ == '__main__':
    app.run(debug=True)