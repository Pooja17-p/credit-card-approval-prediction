from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)
model_path = os.path.join('model', 'c_card_approval_pred.pickle')
with open(model_path, 'rb') as handle:
    model = pickle.load(handle)

@app.route('/')
def home():
    return render_template('ccaindex.html')

@app.route('/Prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('ccaindex1.html')

@app.route('/Home', methods=['GET', 'POST'])
def my_home():
    return render_template('ccaindex.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get values from form, convert to float
        input_features = [float(x) for x in request.form.values()]
        feature_values = [np.array(input_features)]

        # Your column names (MUST match training)
        feature_names = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                         'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                         'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
                         'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'paid_off',
                         '#_of_pastdues', 'no_loan']

        x = pd.DataFrame(feature_values, columns=feature_names)

        pred = model.predict(x)

        prediction = "Eligible" if pred[0] == 1 else "Not Eligible"

        return render_template("Results.html", prediction=prediction)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
