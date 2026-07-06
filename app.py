import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import database
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'credit_card_approval_super_secret_key'

# Initialize database schema on startup
database.init_db()

# Load ML components
MODEL_DIR = 'model'
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
encoder_path = os.path.join(MODEL_DIR, 'encoder.pkl')
model_path = os.path.join(MODEL_DIR, 'model.pkl')

scaler = None
encoder = None
model = None

if os.path.exists(scaler_path) and os.path.exists(encoder_path) and os.path.exists(model_path):
    print("Loading serialized ML components...")
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    model = joblib.load(model_path)
    print("ML components loaded successfully!")
else:
    print("WARNING: Serialized ML components not found. Run train_model.py first.")

# Context Processor for Navbar Auth state
@app.context_processor
def inject_user():
    return dict(current_user=session.get('user'))

# Route: Redirect to Login or Dashboard
@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# Route: Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role', 'User')

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for('register'))

        user_id = database.register_user(name, email, password, role)
        if user_id:
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for('login'))
        else:
            flash("Email already registered.", "danger")
            return redirect(url_for('register'))

    return render_template('register.html')

# Route: Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = database.login_user(email, password)
        if user:
            session['user'] = user
            flash(f"Welcome back, {user['Name']}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password.", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')

# Route: Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# Route: Dashboard
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Fetch stats
    history = database.get_prediction_history()
    models = database.get_models()

    total_predictions = len(history)
    approved_count = sum(1 for p in history if p['ApprovalResult'] == 1)
    rejected_count = total_predictions - approved_count
    
    approval_rate = 0.0
    if total_predictions > 0:
        approval_rate = round((approved_count / total_predictions) * 100, 2)

    # Get active models details
    active_model = None
    for m in models:
        if m['ModelFile'] == 'model.pkl':
            active_model = m

    stats = {
        'total': total_predictions,
        'approved': approved_count,
        'rejected': rejected_count,
        'rate': approval_rate,
        'active_model': active_model
    }

    # Pass the last 5 predictions
    recent_predictions = history[:5]

    return render_template('dashboard.html', stats=stats, recent_predictions=recent_predictions, models=models)

# Route: About
@app.route('/about')
def about():
    if 'user' not in session:
         return redirect(url_for('login'))
    return render_template('about.html')

# Route: Prediction Form and Processing
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Verify model loading status
        if not model or not scaler or not encoder:
            flash("ML prediction pipeline is currently offline. Please contact the administrator.", "danger")
            return redirect(url_for('predict'))

        try:
            # Retrieve applicant attributes
            gender = request.form.get('gender')
            own_car = request.form.get('own_car')
            own_realty = request.form.get('own_realty')
            children = int(request.form.get('children', 0))
            income = float(request.form.get('income', 0))
            income_type = request.form.get('income_type')
            education = request.form.get('education')
            family_status = request.form.get('family_status')
            housing_type = request.form.get('housing_type')
            age = int(request.form.get('age', 30))
            years_employed = float(request.form.get('years_employed', 0))
            work_phone = int(request.form.get('work_phone', 0))
            phone = int(request.form.get('phone', 0))
            email = int(request.form.get('email', 0))
            occupation = request.form.get('occupation', 'Unknown')
            family_members = int(request.form.get('family_members', 1))

            # Feature logic calculations
            is_retired = 1 if (years_employed == 0 and income_type == 'Pensioner') else 0
            # Raw database employed days mapping
            employed_days = 365243 if is_retired else -int(years_employed * 365.25)
            
            # Save applicant profile details to SQLite
            user_id = session['user']['UserID']
            applicant_id = database.add_applicant(
                user_id, income_type, education, family_status, housing_type, 
                age, employed_days, 1, work_phone, phone, email, occupation, family_members
            )

            # Build prediction input DataFrame matching training columns
            input_df = pd.DataFrame([{
                'CODE_GENDER': gender,
                'FLAG_OWN_CAR': own_car,
                'FLAG_OWN_REALTY': own_realty,
                'CNT_CHILDREN': children,
                'AMT_INCOME_TOTAL': income,
                'NAME_INCOME_TYPE': income_type,
                'NAME_EDUCATION_TYPE': education,
                'NAME_FAMILY_STATUS': family_status,
                'NAME_HOUSING_TYPE': housing_type,
                'FLAG_WORK_PHONE': work_phone,
                'FLAG_PHONE': phone,
                'FLAG_EMAIL': email,
                'OCCUPATION_TYPE': occupation,
                'CNT_FAM_MEMBERS': family_members,
                'Age': float(age),
                'Is_Retired': is_retired,
                'Years_Employed': years_employed,
                'Income_Per_Member': float(income) / float(family_members)
            }])

            # Separate columns for Scaling/Encoding
            num_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Age', 'Years_Employed', 'Income_Per_Member']
            cat_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
            bin_cols = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'Is_Retired']

            # Transform features
            input_num_scaled = scaler.transform(input_df[num_cols])
            input_cat_encoded = encoder.transform(input_df[cat_cols])
            input_bin = input_df[bin_cols].values

            # Horizontal stack
            input_proc = np.hstack([input_num_scaled, input_cat_encoded, input_bin])

            # Run inference
            pred_class = int(model.predict(input_proc)[0])
            pred_probs = model.predict_proba(input_proc)[0] if hasattr(model, "predict_proba") else [0.0, 1.0]
            prob_approved = float(pred_probs[1])

            # Logic-based Risk Categories
            if pred_class == 1:
                risk_category = 'Low Risk' if prob_approved >= 0.85 else 'Medium Risk'
            else:
                risk_category = 'High Risk'

            # Fetch active model ID
            active_model_id = None
            models = database.get_models()
            for m in models:
                if m['ModelFile'] == 'model.pkl':
                    active_model_id = m['ModelID']

            # Log prediction result to database
            database.add_prediction(applicant_id, active_model_id, pred_class, risk_category)

            # Construct result payload for templates
            result = {
                'approval_result': pred_class, # 0 or 1
                'risk_category': risk_category,
                'confidence': round(prob_approved * 100, 2) if pred_class == 1 else round((1 - prob_approved) * 100, 2),
                'applicant_name': request.form.get('applicant_name', 'Applicant'),
                'income': income,
                'age': age,
                'decision_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return render_template('result.html', result=result)

        except Exception as e:
            flash(f"Error executing prediction: {str(e)}", "danger")
            return redirect(url_for('predict'))

    return render_template('form.html')

# Route: Prediction History Log list
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_id = None
    if session['user']['Role'] != 'Admin':
        # regular users only see their own predictions
        user_id = session['user']['UserID']

    history_records = database.get_prediction_history(user_id)
    return render_template('history.html', history_records=history_records)

# REST API Endpoint for predicting approval programmatically (e.g. from IBM Cloud/external triggers)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Expects JSON payload
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    if not model or not scaler or not encoder:
        return jsonify({'error': 'Model pipeline offline'}), 503

    try:
        gender = data.get('gender', 'F')
        own_car = data.get('own_car', 'N')
        own_realty = data.get('own_realty', 'Y')
        children = int(data.get('children', 0))
        income = float(data.get('income', 100000))
        income_type = data.get('income_type', 'Working')
        education = data.get('education', 'Secondary / secondary special')
        family_status = data.get('family_status', 'Married')
        housing_type = data.get('housing_type', 'House / apartment')
        age = int(data.get('age', 30))
        years_employed = float(data.get('years_employed', 2.0))
        work_phone = int(data.get('work_phone', 0))
        phone = int(data.get('phone', 0))
        email = int(data.get('email', 0))
        occupation = data.get('occupation', 'Unknown')
        family_members = int(data.get('family_members', 2))

        is_retired = 1 if (years_employed == 0 and income_type == 'Pensioner') else 0

        # Construct DF
        input_df = pd.DataFrame([{
            'CODE_GENDER': gender,
            'FLAG_OWN_CAR': own_car,
            'FLAG_OWN_REALTY': own_realty,
            'CNT_CHILDREN': children,
            'AMT_INCOME_TOTAL': income,
            'NAME_INCOME_TYPE': income_type,
            'NAME_EDUCATION_TYPE': education,
            'NAME_FAMILY_STATUS': family_status,
            'NAME_HOUSING_TYPE': housing_type,
            'FLAG_WORK_PHONE': work_phone,
            'FLAG_PHONE': phone,
            'FLAG_EMAIL': email,
            'OCCUPATION_TYPE': occupation,
            'CNT_FAM_MEMBERS': family_members,
            'Age': float(age),
            'Is_Retired': is_retired,
            'Years_Employed': years_employed,
            'Income_Per_Member': float(income) / float(family_members)
        }])

        num_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Age', 'Years_Employed', 'Income_Per_Member']
        cat_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
        bin_cols = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'Is_Retired']

        input_num_scaled = scaler.transform(input_df[num_cols])
        input_cat_encoded = encoder.transform(input_df[cat_cols])
        input_bin = input_df[bin_cols].values

        input_proc = np.hstack([input_num_scaled, input_cat_encoded, input_bin])

        pred_class = int(model.predict(input_proc)[0])
        pred_probs = model.predict_proba(input_proc)[0] if hasattr(model, "predict_proba") else [0.0, 1.0]
        prob_approved = float(pred_probs[1])

        if pred_class == 1:
            risk_category = 'Low Risk' if prob_approved >= 0.85 else 'Medium Risk'
            confidence = prob_approved
        else:
            risk_category = 'High Risk'
            confidence = 1 - prob_approved

        response = {
            'approval_status': pred_class,
            'status_label': 'Approved' if pred_class == 1 else 'Rejected',
            'risk_category': risk_category,
            'confidence': round(confidence * 100, 2)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
