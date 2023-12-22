from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
import os

dataset_path = os.path.join(os.path.dirname(__file__), 'C:/Users/sriva/project_ml/Hearth-failure-prediction-main/heart_failure_clinical_records_dataset.csv')
df = pd.read_csv(dataset_path)

filename = 'C:/Users/sriva/project_ml/Hearth-failure-prediction-main/model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            cpk = int(request.form['CPK'])
            plate = float(request.form['platelets'])
            sc = float(request.form['SC'])
            ss = int(request.form['SS'])
            ef = int(request.form['EF'])
            time = int(request.form['time'])
            smoke = int(request.form['Smoking'])
            anae = int(request.form['anaemia'])
            pressure = int(request.form['bloodpressure'])
            dia = int(request.form['Diabetes'])
            sex = int(request.form['Gender'])

            # Check for negative values
            if any(val < 0 for val in [age, cpk, plate, sc, ss, ef, time, smoke, anae, pressure, dia, sex]):
                raise ValueError("Input values cannot be negative")

            data = np.array([[age, anae, cpk, dia, ef, pressure, plate, sc, ss, sex, smoke, time]])
            my_prediction = classifier.predict(data)

            if my_prediction == 1:
                prediction_text = "Contact a nearby Heart Doctor"
            else:
                prediction_text = "You are safe"

            return render_template('result.html', prediction_text=prediction_text)
        except ValueError as e:
            # Handle invalid input (non-numeric or negative values)
            error_message = f"Invalid input: {str(e)}"
            return render_template('error.html', error_message=error_message)

    # If the method is not POST, redirect to the home page
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
