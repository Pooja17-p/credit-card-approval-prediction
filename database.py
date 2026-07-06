import os
import sqlite3
import hashlib
from datetime import datetime

DATABASE_NAME = 'credit_card.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON;')
    return conn

def init_db():
    print("Initializing Database...")
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            UserID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            Email TEXT UNIQUE NOT NULL,
            Password TEXT NOT NULL,
            Role TEXT DEFAULT 'User'
        )
    ''')

    # 2. Applicant_Details Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Applicant_Details (
            ApplicantID INTEGER PRIMARY KEY AUTOINCREMENT,
            UserID INTEGER,
            IncomeType TEXT NOT NULL,
            EducationType TEXT NOT NULL,
            FamilyStatus TEXT NOT NULL,
            HousingType TEXT NOT NULL,
            Age INTEGER NOT NULL,
            EmployedDays INTEGER NOT NULL,
            HasMobile INTEGER DEFAULT 1,
            HasWorkPhone INTEGER DEFAULT 0,
            HasPhone INTEGER DEFAULT 0,
            HasEmail INTEGER DEFAULT 0,
            OccupationType TEXT NOT NULL,
            FamilySize INTEGER NOT NULL,
            FOREIGN KEY(UserID) REFERENCES Users(UserID) ON DELETE CASCADE
        )
    ''')

    # 3. Credit_History Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Credit_History (
            HistoryID INTEGER PRIMARY KEY AUTOINCREMENT,
            ApplicantID INTEGER,
            MonthsBalance INTEGER NOT NULL,
            PaymentStatus TEXT NOT NULL,
            OverdueStatus INTEGER DEFAULT 0,
            FOREIGN KEY(ApplicantID) REFERENCES Applicant_Details(ApplicantID) ON DELETE CASCADE
        )
    ''')

    # 4. ML_Model Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ML_Model (
            ModelID INTEGER PRIMARY KEY AUTOINCREMENT,
            ModelName TEXT NOT NULL,
            AlgorithmType TEXT NOT NULL,
            Accuracy REAL NOT NULL,
            ModelFile TEXT NOT NULL
        )
    ''')

    # 5. Approval_Prediction Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Approval_Prediction (
            PredictionID INTEGER PRIMARY KEY AUTOINCREMENT,
            ApplicantID INTEGER,
            ModelID INTEGER,
            ApprovalResult INTEGER NOT NULL, -- 0 = Rejected, 1 = Approved
            RiskCategory TEXT NOT NULL,       -- Low Risk, Medium Risk, High Risk
            PredictionDate TEXT NOT NULL,
            FOREIGN KEY(ApplicantID) REFERENCES Applicant_Details(ApplicantID) ON DELETE CASCADE,
            FOREIGN KEY(ModelID) REFERENCES ML_Model(ModelID) ON DELETE SET NULL
        )
    ''')

    # Seed ML Model metadata if empty
    cursor.execute("SELECT COUNT(*) FROM ML_Model")
    if cursor.fetchone()[0] == 0:
        print("Seeding trained models metadata...")
        models_metadata = [
            ('Random Forest Classifier', 'Random Forest', 0.7460, 'model.pkl'),
            ('Logistic Regression Classifier', 'Logistic Regression', 0.7280, 'logistic_regression.pkl'),
            ('Decision Tree Classifier', 'Decision Tree', 0.7210, 'decision_tree.pkl'),
            ('XGBoost Classifier', 'XGBoost', 0.7070, 'xgboost.pkl')
        ]
        cursor.executemany('''
            INSERT INTO ML_Model (ModelName, AlgorithmType, Accuracy, ModelFile)
            VALUES (?, ?, ?, ?)
        ''', models_metadata)

    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User Auth Functions
def register_user(name, email, password, role='User'):
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_pwd = hash_password(password)
    try:
        cursor.execute('''
            INSERT INTO Users (Name, Email, Password, Role)
            VALUES (?, ?, ?, ?)
        ''', (name, email, hashed_pwd, role))
        conn.commit()
        user_id = cursor.lastrowid
        print(f"Registered user: {email} with ID: {user_id}")
        return user_id
    except sqlite3.IntegrityError:
        print(f"Registration failed: Email {email} already exists.")
        return None
    finally:
        conn.close()

def login_user(email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_pwd = hash_password(password)
    cursor.execute('''
        SELECT UserID, Name, Email, Role FROM Users
        WHERE Email = ? AND Password = ?
    ''', (email, hashed_pwd))
    user = cursor.fetchone()
    conn.close()
    if user:
        return dict(user)
    return None

# Add Applicant Profile
def add_applicant(user_id, income_type, education_type, family_status, housing_type, age, employed_days, has_mobile, has_work_phone, has_phone, has_email, occupation_type, family_size):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Applicant_Details (
            UserID, IncomeType, EducationType, FamilyStatus, HousingType, 
            Age, EmployedDays, HasMobile, HasWorkPhone, HasPhone, HasEmail, 
            OccupationType, FamilySize
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, income_type, education_type, family_status, housing_type, age, employed_days, has_mobile, has_work_phone, has_phone, has_email, occupation_type, family_size))
    conn.commit()
    applicant_id = cursor.lastrowid
    conn.close()
    return applicant_id

# Add Prediction log
def add_prediction(applicant_id, model_id, approval_result, risk_category):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO Approval_Prediction (
            ApplicantID, ModelID, ApprovalResult, RiskCategory, PredictionDate
        ) VALUES (?, ?, ?, ?, ?)
    ''', (applicant_id, model_id, approval_result, risk_category, now_str))
    conn.commit()
    prediction_id = cursor.lastrowid
    conn.close()
    return prediction_id

# Retrieve all prediction history (with joins for UI tables)
def get_prediction_history(user_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = '''
        SELECT 
            ap.PredictionID, 
            ap.ApprovalResult, 
            ap.RiskCategory, 
            ap.PredictionDate,
            ad.Age, 
            ad.IncomeType, 
            ad.EducationType, 
            ad.OccupationType,
            m.ModelName,
            u.Name as AnalystName
        FROM Approval_Prediction ap
        JOIN Applicant_Details ad ON ap.ApplicantID = ad.ApplicantID
        JOIN Users u ON ad.UserID = u.UserID
        LEFT JOIN ML_Model m ON ap.ModelID = m.ModelID
    '''
    if user_id:
        query += " WHERE ad.UserID = ?"
        cursor.execute(query + " ORDER BY ap.PredictionDate DESC", (user_id,))
    else:
        cursor.execute(query + " ORDER BY ap.PredictionDate DESC")
        
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Retrieve models from DB
def get_models():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ML_Model")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_model_by_id(model_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ML_Model WHERE ModelID = ?", (model_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

if __name__ == "__main__":
    init_db()
