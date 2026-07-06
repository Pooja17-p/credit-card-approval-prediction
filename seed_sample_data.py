import os
import sqlite3
from datetime import datetime, timedelta
import database

def seed_data():
    print("Seeding sample data for verification...")
    # Ensure database is initialized
    database.init_db()

    # 1. Check if already seeded (isolated connection)
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Users WHERE Email = 'admin@bank.com'")
    already_seeded = cursor.fetchone()[0] > 0
    conn.close()

    if already_seeded:
        print("Sample data already seeded. Skipping.")
        return

    # 2. Seed Analyst Users (uses internal database.py connections)
    admin_id = database.register_user("Charles Vance", "admin@bank.com", "admin123", "Admin")
    analyst_id = database.register_user("Clara Oswald", "clara@bank.com", "clara123", "User")
    print(f"Seeded Users: admin_id={admin_id}, analyst_id={analyst_id}")

    # 3. Seed Applicants and Predictions (uses a single isolated connection)
    conn = database.get_db_connection()
    cursor = conn.cursor()

    # Get active model ID
    cursor.execute("SELECT ModelID FROM ML_Model WHERE ModelFile = 'model.pkl'")
    model_row = cursor.fetchone()
    model_id = model_row[0] if model_row else 1

    sample_applicants = [
        # Applicant 1: Low Risk, Approved
        {
            'user_id': admin_id, 'income_type': 'Working', 'education_type': 'Higher education',
            'family_status': 'Married', 'housing_type': 'House / apartment', 'age': 34,
            'employed_days': -1500, 'occupation_type': 'Managers', 'family_size': 3,
            'approval': 1, 'risk': 'Low Risk', 'days_ago': 5
        },
        # Applicant 2: High Risk, Rejected
        {
            'user_id': analyst_id, 'income_type': 'Working', 'education_type': 'Secondary / secondary special',
            'family_status': 'Single / not married', 'housing_type': 'Rented apartment', 'age': 24,
            'employed_days': -120, 'occupation_type': 'Laborers', 'family_size': 1,
            'approval': 0, 'risk': 'High Risk', 'days_ago': 4
        },
        # Applicant 3: Medium Risk, Approved
        {
            'user_id': analyst_id, 'income_type': 'Commercial associate', 'education_type': 'Higher education',
            'family_status': 'Civil marriage', 'housing_type': 'House / apartment', 'age': 41,
            'employed_days': -800, 'occupation_type': 'Sales staff', 'family_size': 2,
            'approval': 1, 'risk': 'Medium Risk', 'days_ago': 3
        },
        # Applicant 4: Low Risk, Approved
        {
            'user_id': admin_id, 'income_type': 'Pensioner', 'education_type': 'Secondary / secondary special',
            'family_status': 'Widow', 'housing_type': 'House / apartment', 'age': 62,
            'employed_days': 365243, 'occupation_type': 'Unknown', 'family_size': 1,
            'approval': 1, 'risk': 'Low Risk', 'days_ago': 2
        },
        # Applicant 5: High Risk, Rejected
        {
            'user_id': admin_id, 'income_type': 'Working', 'education_type': 'Secondary / secondary special',
            'family_status': 'Separated', 'housing_type': 'With parents', 'age': 29,
            'employed_days': -300, 'occupation_type': 'Drivers', 'family_size': 2,
            'approval': 0, 'risk': 'High Risk', 'days_ago': 1
        }
    ]

    for app in sample_applicants:
        # Add Applicant directly using our current cursor
        cursor.execute('''
            INSERT INTO Applicant_Details (
                UserID, IncomeType, EducationType, FamilyStatus, HousingType, 
                Age, EmployedDays, HasMobile, HasWorkPhone, HasPhone, HasEmail, 
                OccupationType, FamilySize
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (app['user_id'], app['income_type'], app['education_type'], app['family_status'], app['housing_type'],
              app['age'], app['employed_days'], 1, 0, 1, 0, app['occupation_type'], app['family_size']))
        
        app_id = cursor.lastrowid
        
        # Add Prediction directly using our current cursor
        date_str = (datetime.now() - timedelta(days=app['days_ago'])).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO Approval_Prediction (ApplicantID, ModelID, ApprovalResult, RiskCategory, PredictionDate)
            VALUES (?, ?, ?, ?, ?)
        ''', (app_id, model_id, app['approval'], app['risk'], date_str))

    conn.commit()
    conn.close()
    print("Database seeded with sample records successfully!")

if __name__ == "__main__":
    seed_data()
