import os
import json
import requests
import joblib
import numpy as np
import pandas as pd

# IBM Cloud and Watson Machine Learning Configurations
# Replace these with your actual credentials when deploying to IBM Cloud
API_KEY = "YOUR_IBM_CLOUD_API_KEY"
WML_URL = "https://us-south.ml.cloud.ibm.com"  # e.g., us-south, eu-de, jp-tok, etc.
SPACE_ID = "YOUR_WATSON_ML_SPACE_ID"
DEPLOYMENT_ID = "YOUR_WATSON_ML_DEPLOYMENT_ID"

class IBMWatsonMLClient:
    """
    Helper class to manage communication with IBM Watson Machine Learning REST APIs
    without requiring heavy dependencies, ensuring 100% compatibility.
    """
    def __init__(self, api_key=API_KEY, wml_url=WML_URL, space_id=SPACE_ID, deployment_id=DEPLOYMENT_ID):
        self.api_key = api_key
        self.wml_url = wml_url
        self.space_id = space_id
        self.deployment_id = deployment_id
        self.token = None

    def get_iam_token(self):
        """
        Retrieves the OAuth 2.0 Token from IBM Cloud IAM service.
        """
        print("Authenticating with IBM Cloud IAM...")
        if self.api_key == "YOUR_IBM_CLOUD_API_KEY":
            print("WARNING: Using placeholder API key. Authentication will fail.")
            return None

        url = "https://iam.cloud.ibm.com/identity/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.api_key
        }

        try:
            response = requests.post(url, headers=headers, data=data)
            if response.status_code == 200:
                self.token = response.json().get("access_token")
                print("Successfully authenticated and received IAM Token!")
                return self.token
            else:
                print(f"Authentication failed (HTTP {response.status_code}): {response.text}")
                return None
        except Exception as e:
            print(f"Error during authentication: {str(e)}")
            return None

    def score_model(self, preprocessed_record):
        """
        Sends the preprocessed record to the deployed Watson ML endpoint for scoring.
        
        Parameters:
            preprocessed_record (np.ndarray): 1D array of 56 processed features.
        """
        if not self.token:
            self.get_iam_token()

        if not self.token:
            print("Cannot score model: Missing IAM Token.")
            return None

        # WML Scoring endpoint URL format
        scoring_url = f"{self.wml_url}/ml/v4/deployments/{self.deployment_id}/predictions?version=2021-05-01"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
            "ML-Instance-ID": self.space_id
        }

        # Convert numpy array to standard list of lists for JSON serialization
        values_list = preprocessed_record.tolist()
        if not isinstance(values_list[0], list):
            values_list = [values_list]  # Ensure it is a list of lists

        # Watson ML expects payload format: {"input_data": [{"values": [[...]]}]}
        payload = {
            "input_data": [
                {
                    "values": values_list
                }
            ]
        }

        try:
            print(f"Sending scoring request to Watson ML deployment: {self.deployment_id}...")
            response = requests.post(scoring_url, json=payload, headers=headers)
            if response.status_code == 200:
                print("Scoring successful!")
                return response.json()
            else:
                print(f"Scoring request failed (HTTP {response.status_code}): {response.text}")
                return None
        except Exception as e:
            print(f"Error calling scoring API: {str(e)}")
            return None

def demo_cloud_prediction_flow():
    """
    Demonstrates the end-to-end pipeline:
    1. Loads a test applicant record
    2. Runs local pre-processing (scaling & encoding)
    3. Packages the record for IBM Watson Machine Learning format
    """
    print("\n=== DEMO CLOUD PREDICTION FLOW ===")
    
    # 1. Load Preprocessors
    if not os.path.exists('model/scaler.pkl') or not os.path.exists('model/encoder.pkl'):
        print("Please run train_model.py first to generate the preprocessors.")
        return
        
    scaler = joblib.load('model/scaler.pkl')
    encoder = joblib.load('model/encoder.pkl')
    
    # 2. Simulate an applicant data
    applicant_data = {
        'CODE_GENDER': 'F',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'Y',
        'CNT_CHILDREN': 1,
        'AMT_INCOME_TOTAL': 150000.0,
        'NAME_INCOME_TYPE': 'Working',
        'NAME_EDUCATION_TYPE': 'Higher education',
        'NAME_FAMILY_STATUS': 'Married',
        'NAME_HOUSING_TYPE': 'House / apartment',
        'FLAG_WORK_PHONE': 0,
        'FLAG_PHONE': 1,
        'FLAG_EMAIL': 0,
        'OCCUPATION_TYPE': 'Core staff',
        'CNT_FAM_MEMBERS': 3,
        'Age': 32.5,
        'Is_Retired': 0,
        'Years_Employed': 4.0,
        'Income_Per_Member': 150000.0 / 3
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([applicant_data])
    
    # Separate features for preprocessors
    num_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Age', 'Years_Employed', 'Income_Per_Member']
    cat_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
    bin_cols = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'Is_Retired']
    
    num_scaled = scaler.transform(df[num_cols])
    cat_encoded = encoder.transform(df[cat_cols])
    bin_features = df[bin_cols].values
    
    # Merge preprocessed features
    preprocessed_features = np.hstack([num_scaled, cat_encoded, bin_features])
    print(f"Preprocessed features shape: {preprocessed_features.shape}")
    
    # 3. Build Watson ML scoring payload schema
    payload = {
        "input_data": [
            {
                "fields": [f"f_{i}" for i in range(preprocessed_features.shape[1])],
                "values": preprocessed_features.tolist()
            }
        ]
    }
    
    print("\nPrepared Watson Machine Learning JSON payload (truncated):")
    print(json.dumps(payload, indent=2)[:500] + "\n... [truncated] ...")
    
    # 4. Initialize client
    wml_client = IBMWatsonMLClient()
    print("\nWML Client initialized. Ready to swap in credentials and call score_model().")
    print("===================================\n")

if __name__ == "__main__":
    demo_cloud_prediction_flow()
