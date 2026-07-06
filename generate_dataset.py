import os
import pandas as pd
import numpy as np

def generate_data(num_clients=5000, seed=42):
    np.random.seed(seed)
    print(f"Generating synthetic Credit Card dataset with {num_clients} clients...")

    # Ensure dataset directory exists
    os.makedirs('dataset', exist_ok=True)

    # 1. Generate Application Records (application_record.csv)
    # Generate unique 7-digit IDs
    client_ids = np.random.choice(np.arange(5000000, 6000000), size=num_clients, replace=False)

    genders = np.random.choice(['M', 'F'], size=num_clients, p=[0.38, 0.62])
    own_car = np.random.choice(['Y', 'N'], size=num_clients, p=[0.35, 0.65])
    own_realty = np.random.choice(['Y', 'N'], size=num_clients, p=[0.69, 0.31])
    
    # Children count: mostly 0, 1, or 2, rarely more
    children = np.random.choice([0, 1, 2, 3, 4, 5], size=num_clients, p=[0.70, 0.18, 0.09, 0.02, 0.007, 0.003])
    
    # Total Income: Log-normal distribution to mimic actual incomes
    income = np.random.lognormal(mean=12.0, sigma=0.5, size=num_clients)
    # Round to nearest 500
    income = np.round(income / 500) * 500
    
    income_types = np.random.choice(
        ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'],
        size=num_clients,
        p=[0.51, 0.23, 0.17, 0.088, 0.002]
    )
    
    education_types = np.random.choice(
        ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'],
        size=num_clients,
        p=[0.71, 0.25, 0.03, 0.009, 0.001]
    )
    
    family_statuses = np.random.choice(
        ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'],
        size=num_clients,
        p=[0.69, 0.14, 0.08, 0.06, 0.03]
    )
    
    housing_types = np.random.choice(
        ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment'],
        size=num_clients,
        p=[0.89, 0.045, 0.03, 0.018, 0.013, 0.004]
    )
    
    # Days birth: between 21 and 68 years old (expressed as negative days)
    age_years = np.random.uniform(21, 68, size=num_clients)
    days_birth = -np.round(age_years * 365.25).astype(int)
    
    # Days employed: negative days, pensioners mostly have positive 365243 (meaning unemployed/retired)
    days_employed = []
    for i in range(num_clients):
        if income_types[i] == 'Pensioner':
            # 85% of pensioners are retired
            if np.random.rand() < 0.85:
                days_employed.append(365243)
                continue
        # Employed
        emp_years = np.random.uniform(0.1, age_years[i] - 18)  # started working after 18
        days_employed.append(-int(np.round(emp_years * 365.25)))
    days_employed = np.array(days_employed)
    
    # Flags
    flag_mobil = np.ones(num_clients, dtype=int)  # everyone has a mobile
    flag_work_phone = np.random.choice([0, 1], size=num_clients, p=[0.78, 0.22])
    flag_phone = np.random.choice([0, 1], size=num_clients, p=[0.70, 0.30])
    flag_email = np.random.choice([0, 1], size=num_clients, p=[0.91, 0.09])
    
    # Occupation types
    occupations = [
        'Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
        'High skill tech staff', 'Accountants', 'Medicine staff', 'Cooking staff', 
        'Security staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers', 
        'Waiters/barmen staff', 'Secretaries', 'HR staff', 'Realty agents', 'IT staff'
    ]
    occupation_p = [
        0.30, 0.16, 0.15, 0.14, 0.10, 0.04, 0.03, 0.025, 0.015, 
        0.013, 0.010, 0.005, 0.003, 0.003, 0.003, 0.002, 0.002, 0.001
    ]
    occupation_p = np.array(occupation_p)
    occupation_p = occupation_p / occupation_p.sum()
    
    occupation_type = []
    for i in range(num_clients):
        if income_types[i] == 'Pensioner' and days_employed[i] == 365243:
            occupation_type.append(np.nan)  # retired people don't have active occupations
        else:
            # Introduce 30% missing values in occupation type for employed people to simulate missing value handling
            if np.random.rand() < 0.30:
                occupation_type.append(np.nan)
            else:
                occupation_type.append(np.random.choice(occupations, p=occupation_p))
                
    # Family size: child count + (1 or 2 depending on marital status)
    family_size = []
    for i in range(num_clients):
        spouse = 2 if family_statuses[i] in ['Married', 'Civil marriage'] else 1
        family_size.append(children[i] + spouse)
    family_size = np.array(family_size)
    
    app_df = pd.DataFrame({
        'ID': client_ids,
        'CODE_GENDER': genders,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_realty,
        'CNT_CHILDREN': children,
        'AMT_INCOME_TOTAL': income,
        'NAME_INCOME_TYPE': income_types,
        'NAME_EDUCATION_TYPE': education_types,
        'NAME_FAMILY_STATUS': family_statuses,
        'NAME_HOUSING_TYPE': housing_types,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'FLAG_MOBIL': flag_mobil,
        'FLAG_WORK_PHONE': flag_work_phone,
        'FLAG_PHONE': flag_phone,
        'FLAG_EMAIL': flag_email,
        'OCCUPATION_TYPE': occupation_type,
        'CNT_FAM_MEMBERS': family_size
    })

    # Add duplicate rows to test duplicate cleaning (Task 10)
    dup_indices = np.random.choice(num_clients, size=25, replace=False)
    dup_rows = app_df.iloc[dup_indices].copy()
    # Modify IDs slightly for some, but keep some identical
    app_df = pd.concat([app_df, dup_rows], ignore_index=True)
    
    # 2. Generate Credit Records (credit_record.csv)
    # Each client will have history of 6 to 60 months
    credit_records = []
    for cid in client_ids:
        num_months = np.random.randint(6, 61)
        start_month = -num_months + 1
        
        # Determine risk profile of applicant based on income, employment, and age
        # This creates a realistic correlation between demographic features and approval status
        risk_score = 0.0
        # Income effect (higher income = lower risk)
        idx = np.where(client_ids == cid)[0][0]
        inc = income[idx]
        if inc < 100000:
            risk_score += 0.4
        elif inc < 180000:
            risk_score += 0.2
            
        # Employment days effect (longer employment = lower risk)
        emp_days = days_employed[idx]
        if emp_days == 365243:  # Pensioner / Unemployed
            risk_score += 0.25
        elif emp_days > -365 * 2:  # Less than 2 years employed
            risk_score += 0.2
            
        # Education effect (higher education = lower risk)
        edu = education_types[idx]
        if edu == 'Secondary / secondary special':
            risk_score += 0.15
        elif edu == 'Lower secondary':
            risk_score += 0.3
            
        # Random variance
        risk_score += np.random.uniform(-0.2, 0.3)
        
        # Payment behaviors:
        # C: paid off, X: no loan, 0: 1-29 days overdue, 1: 30-59 days, 2: 60-89 days, 3: 90-119 days, 4: 120-149 days, 5: 150+ days overdue
        for m in range(start_month, 1):
            if risk_score > 0.65:
                # High risk: higher chance of late payments (0, 1, 2, 3, 4, 5)
                status = np.random.choice(
                    ['C', 'X', '0', '1', '2', '3', '4', '5'],
                    p=[0.25, 0.15, 0.35, 0.12, 0.06, 0.04, 0.02, 0.01]
                )
            elif risk_score > 0.35:
                # Medium risk: occasional late payments
                status = np.random.choice(
                    ['C', 'X', '0', '1', '2', '3', '4', '5'],
                    p=[0.40, 0.20, 0.32, 0.05, 0.02, 0.007, 0.002, 0.001]
                )
            else:
                # Low risk: clean payment history
                status = np.random.choice(
                    ['C', 'X', '0', '1'],
                    p=[0.60, 0.25, 0.145, 0.005]
                )
            credit_records.append({
                'ID': cid,
                'MONTHS_BALANCE': m,
                'STATUS': status
            })
            
    credit_df = pd.DataFrame(credit_records)

    # Save to disk
    app_df.to_csv('dataset/application_record.csv', index=False)
    credit_df.to_csv('dataset/credit_record.csv', index=False)
    print("Dataset generated successfully!")
    print(f"Saved application records: {len(app_df)} rows to dataset/application_record.csv")
    print(f"Saved credit records: {len(credit_df)} rows to dataset/credit_record.csv")

if __name__ == "__main__":
    generate_data()
