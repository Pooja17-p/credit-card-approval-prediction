import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def run_preprocessing_and_eda():
    print("-----------------------------------------")
    print("PHASE 2: DATA PREPROCESSING & EDA STARTED")
    print("-----------------------------------------")

    # Create directories if not existing
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)

    # Task 6: Read Dataset
    print("\n[Task 6] Reading datasets...")
    app_df = pd.read_csv('dataset/application_record.csv')
    credit_df = pd.read_csv('dataset/credit_record.csv')
    print(f"Application record shape: {app_df.shape}")
    print(f"Credit record shape: {credit_df.shape}")

    # Task 7: Descriptive Analysis
    print("\n[Task 7] Descriptive Analysis:")
    print("Application Dataset Info Summary:")
    print(app_df.info())
    print("\nApplication Dataset Numerical Summary:")
    print(app_df.describe().T)
    print("\nApplication Dataset Null Count:")
    print(app_df.isnull().sum())

    # Task 10: Drop Duplicate Features/Rows
    print("\n[Task 10] Handling duplicates...")
    print(f"Duplicates before cleaning: {app_df.duplicated(subset=['ID']).sum()}")
    app_df = app_df.drop_duplicates(subset=['ID'], keep='first')
    print(f"Application record shape after deduplication: {app_df.shape}")

    # Task 11: Handle Missing Values
    print("\n[Task 11] Handling missing values...")
    # Missing values in OCCUPATION_TYPE
    app_df['OCCUPATION_TYPE'] = app_df['OCCUPATION_TYPE'].fillna('Unknown')
    print("Null counts after handling:")
    print(app_df.isnull().sum())

    # Task 12: Data Cleaning and Merging (Vintage/Payment History Labeling)
    print("\n[Task 12] Label Generation via Credit Vintage Analysis...")
    # Label mapping rules:
    # Status values:
    # 0: 1-29 days overdue
    # 1: 30-59 days overdue
    # 2: 60-89 days overdue
    # 3: 90-119 days overdue
    # 4: 120-149 days overdue
    # 5: 150+ days overdue (bad debt/write-offs)
    # C: paid off that month
    # X: No loan / no activity
    # Rules: Applicants with 60+ days overdue (status 2, 3, 4, 5) are flagged as High Risk (0 = Rejected)
    # Applicants with status C, X, 0, or 1 are flagged as Low Risk (1 = Approved)
    
    # Check max status for each user
    # Map statuses to numeric risk weights
    status_map = {
        'X': -1,
        'C': -1,
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5
    }
    credit_df['STATUS_NUM'] = credit_df['STATUS'].map(status_map)
    
    # Find max status level per user
    user_max_status = credit_df.groupby('ID')['STATUS_NUM'].max().reset_index()
    
    # Label construction: Approved (1) if max status <= 1, else Rejected (0)
    user_max_status['Approval_Status'] = np.where(user_max_status['STATUS_NUM'] >= 2, 0, 1)
    
    print("Class Distribution in target variable:")
    print(user_max_status['Approval_Status'].value_counts())
    print("Target status percentage:")
    print(user_max_status['Approval_Status'].value_counts(normalize=True) * 100)

    # Merge application dataset with target labels
    print("Merging datasets on 'ID'...")
    merged_df = pd.merge(app_df, user_max_status[['ID', 'Approval_Status']], on='ID', how='inner')
    print(f"Merged Dataset Shape: {merged_df.shape}")

    # Task 13: Feature Engineering
    print("\n[Task 13] Feature Engineering...")
    # Convert DAYS_BIRTH (negative) to AGE (years)
    merged_df['Age'] = -merged_df['DAYS_BIRTH'] / 365.25
    
    # Convert DAYS_EMPLOYED to employment duration
    # Code 365243 means unemployed/retired
    merged_df['Is_Retired'] = np.where(merged_df['DAYS_EMPLOYED'] == 365243, 1, 0)
    merged_df['Years_Employed'] = np.where(merged_df['DAYS_EMPLOYED'] == 365243, 0, -merged_df['DAYS_EMPLOYED'] / 365.25)
    
    # Income per family member
    merged_df['Income_Per_Member'] = merged_df['AMT_INCOME_TOTAL'] / merged_df['CNT_FAM_MEMBERS']

    # Dropping original columns that have been transformed
    merged_df = merged_df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL'])
    print(f"Dataset columns after engineering: {list(merged_df.columns)}")

    # Task 8 & 9: Univariate & Multivariate Visualization
    print("\n[Task 8 & 9] Generating visualization charts...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Income distribution (Univariate)
    plt.figure(figsize=(8, 5))
    sns.histplot(merged_df['AMT_INCOME_TOTAL'], kde=True, color='#2ec4b6', bins=30)
    plt.title('Income Distribution of Applicants', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Annual Income ($)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('static/images/income_distribution.png', dpi=300)
    plt.close()

    # 2. Age distribution (Univariate)
    plt.figure(figsize=(8, 5))
    sns.histplot(merged_df['Age'], kde=True, color='#e71d36', bins=30)
    plt.title('Age Distribution of Applicants', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Age (Years)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('static/images/age_distribution.png', dpi=300)
    plt.close()

    # 3. Correlation Heatmap (Multivariate)
    plt.figure(figsize=(10, 8))
    numeric_cols = merged_df.select_dtypes(include=[np.number]).drop(columns=['ID'])
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('static/images/correlation_heatmap.png', dpi=300)
    plt.close()

    # 4. Income Type vs Approval Status (Multivariate)
    plt.figure(figsize=(9, 5))
    sns.countplot(data=merged_df, x='NAME_INCOME_TYPE', hue='Approval_Status', palette='Set2')
    plt.title('Approval Status by Income Type', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Income Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=15)
    plt.legend(title='Approval Status', labels=['Rejected (0)', 'Approved (1)'])
    plt.tight_layout()
    plt.savefig('static/images/income_type_approval.png', dpi=300)
    plt.close()
    print("Visualization charts saved to static/images/")

    # Task 14: Handle Categorical Variables & Train-Test Split
    print("\n[Task 14] Splitting and preparing data...")
    
    # Save the clean merged dataframe before encoding for reference
    merged_df.to_csv('dataset/cleaned_merged_data.csv', index=False)
    print("Clean merged dataset saved to dataset/cleaned_merged_data.csv")
    
    # Define features and target
    X = merged_df.drop(columns=['ID', 'Approval_Status'])
    y = merged_df['Approval_Status']

    # Train-test split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Save split datasets
    X_train.to_csv('dataset/X_train.csv', index=False)
    X_test.to_csv('dataset/X_test.csv', index=False)
    y_train.to_csv('dataset/y_train.csv', index=False)
    y_test.to_csv('dataset/y_test.csv', index=False)
    
    print("Preprocessing and EDA completed successfully!")

if __name__ == "__main__":
    run_preprocessing_and_eda()
