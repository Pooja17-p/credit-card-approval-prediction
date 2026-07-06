# Credit Card Approval Prediction Using IBM Watson Machine Learning

An end-to-end, industry-grade credit risk underwriting application that classifies credit card applicants as **Approved (1)** or **Rejected (0)** using Machine Learning and a secure, multi-user Flask dashboard. Developed as part of the SmartBridge Artificial Intelligence & Machine Learning Program.

---

## Table of Contents
1. [Project Objective](#project-objective)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Project Architecture](#project-architecture)
4. [Database Design](#database-design)
5. [Application Flow](#application-flow)
6. [Folder Structure](#folder-structure)
7. [Machine Learning Performance Comparison](#machine-learning-performance-comparison)
8. [IBM Watson ML Cloud Deployment](#ibm-watson-ml-cloud-deployment)
9. [Running the Application](#running-the-application)
10. [Future Scope & Conclusion](#future-scope--conclusion)

---

## Project Objective
The objective of this project is to develop a secure, highly responsive, and robust web application for credit analysts and underwriters. The system automates credit card risk classification by:
- Collecting applicant socio-demographic details and assets.
- Cleaning, preprocessing, and engineering features (e.g., converting dates to age, calculating income metrics).
- Applying pre-trained machine learning classifiers to score applications.
- Cataloging transactions and histories inside a normalized SQLite database.
- Structuring APIs for deployment to IBM Watson Machine Learning.

---

## Prerequisites & Installation

### System Requirements
- Python 3.8 to 3.14
- SQLite 3
- Modern Web Browser (Chrome, Firefox, Edge, Safari)

### Installation Guide
1. **Clone the repository / Enter directory**:
   ```bash
   cd c:\CreditCardApprovalPrediction
   ```

2. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Architecture

The platform is designed around a **5-Tier Layered Architecture**:

1. **Presentation Layer (Web UI)**:
   - Built with responsive HTML5, Bootstrap 5, and custom Glassmorphism CSS styling.
   - Includes user authentication forms, scoring submission grids, loading overlays, and dynamic dashboards featuring statistical charts.

2. **Application Layer (Flask Controller)**:
   - Uses `app.py` as the backend controller.
   - Manages user sessions, route mappings, form input parsing, and data prep transformations.

3. **Machine Learning Layer**:
   - Contains model training (`train_model.py`) and preprocessing scripts (`preprocessing_eda.py`).
   - Implements scaling (`scaler.pkl`), encoding (`encoder.pkl`), and trained classifiers (`model.pkl`).

4. **Database Layer (SQLite)**:
   - Uses `database.py` to create and query a normalized SQLite relational database (`credit_card.db`).
   - Manages tables: `Users` (with SHA-256 hashed credentials), `Applicant_Details`, `Credit_History`, `ML_Model`, and `Approval_Prediction`.

5. **Deployment Layer (IBM Cloud / Watson ML)**:
   - Connects to IBM Cloud IAM and Watson Machine Learning REST APIs.
   - Uses the helper scripts in `wml_deploy.py` to authenticate and submit scoring payloads to Watson ML deployments in the cloud.

---

## Database Design

The relational database is normalized and enforces primary/foreign key relationships.

```
+-----------------------------------------------------------------+
|                              USERS                              |
| (UserID PK, Name, Email UNIQUE, Password, Role)                  |
+-----------------------------------------------------------------+
                                | 1
                                |
                                | N
+-----------------------------------------------------------------+
|                       APPLICANT_DETAILS                         |
| (ApplicantID PK, UserID FK, IncomeType, EducationType,          |
|  FamilyStatus, HousingType, Age, EmployedDays, HasMobile,       |
|  HasWorkPhone, HasPhone, HasEmail, OccupationType, FamilySize)  |
+-----------------------------------------------------------------+
            | 1                                       | 1
            |                                         |
            | N                                       | 1
+-----------------------+                 +-----------------------+
|    CREDIT_HISTORY     |                 |  APPROVAL_PREDICTION  |
| (HistoryID PK,        |                 | (PredictionID PK,     |
|  ApplicantID FK,      |                 |  ApplicantID FK,      |
|  MonthsBalance,       |                 |  ModelID FK,          |
|  PaymentStatus,       |                 |  ApprovalResult,      |
|  OverdueStatus)       |                 |  RiskCategory,        |
+-----------------------+                 |  PredictionDate)      |
                                          +-----------------------+
                                                      | N
                                                      |
                                                      | 1
                                          +-----------------------+
                                          |       ML_MODEL        |
                                          | (ModelID PK, Name,    |
                                          |  Algorithm, Accuracy, |
                                          |  ModelFile)           |
                                          +-----------------------+
```

---

## Application Flow

```
   User
    │
    ▼
  Login ────► Session Authenticated
    │
    ▼
Dashboard ──► Click "New Score Request"
    │
    ▼
Applicant Form (Enter socio-demographic details)
    │
    ▼
  Submit ───► Flask Backend Controller (app.py)
                │
                ├─► Data Validation & Preprocessing (Scale & Encode)
                ├─► Database Write (Log Applicant Details)
                ├─► Model Inference (Predict Class & Probability)
                ├─► Risk Classification (Low / Medium / High Risk)
                └─► Database Write (Log Prediction Ledger)
    │
    ▼
Result Page (Animated status badge, risk gauge, confidence bar)
    │
    ▼
History Page (Audit log of all decisions)
```

---

## Folder Structure

```
CreditCardApprovalPrediction/
│
├── dataset/                     # Data source files
│   ├── application_record.csv   # Raw applicant records (generated)
│   ├── credit_record.csv        # Raw payment history records (generated)
│   ├── cleaned_merged_data.csv  # Merged and labeled dataset (vintage analysis)
│   ├── X_train.csv / y_train.csv# 80% training features and labels
│   └── X_test.csv / y_test.csv  # 20% test features and labels
│
├── model/                       # Serialized machine learning binaries
│   ├── model.pkl                # Selected best classifier (Random Forest)
│   ├── scaler.pkl               # Standard scaler preprocessing model
│   └── encoder.pkl              # One-Hot categorical encoder
│
├── notebooks/                   # Preprocessing and analysis notebooks
│   └── preprocessing_eda.py     # Data cleaning, feature engineering & EDA script
│
├── templates/                   # Flask HTML layout templates
│   ├── base.html                # Master skeleton layout
│   ├── login.html               # Authentication login page
│   ├── register.html            # Registration page
│   ├── dashboard.html           # Analytical stats dashboard
│   ├── form.html                # Application input page
│   ├── result.html              # Model prediction score report
│   ├── history.html             # Audit history table
│   └── about.html               # Layered design details page
│
├── static/                      # Static assets served by web app
│   ├── css/
│   │   └── styles.css           # Custom Glassmorphic stylesheet
│   ├── js/
│   │   └── main.js              # Validations & animated loading scripts
│   └── images/
│       ├── income_distribution.png # EDA plot
│       ├── age_distribution.png    # EDA plot
│       ├── correlation_heatmap.png # EDA plot
│       ├── income_type_approval.png# EDA plot
│       ├── model_comparison.png    # Classifier comparison metrics plot
│       └── cm_*.png                # Individual confusion matrix plots
│
├── app.py                       # Core Flask application controller
├── train_model.py               # Model training & serialization pipeline
├── database.py                  # SQLite database manager & CRUD operations
├── wml_deploy.py                # IBM Watson ML REST API deployment client
├── requirements.txt             # Python packages manifest
└── README.md                    # System documentation
```

---

## Machine Learning Performance Comparison

Four classification models were trained and evaluated on the preprocessed 20% test set (1,000 samples). Below is their performance:

| Classifier Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 72.80% | 74.87% | 88.30% | 81.03% | 0.7769 |
| **Decision Tree** | 72.10% | 75.78% | 84.65% | 79.97% | 0.7371 |
| **Random Forest** | **74.60%** | **75.70%** | **90.43%** | **82.41%** | **0.7913** |
| **XGBoost** | 70.70% | 75.38% | 82.37% | 78.72% | 0.7569 |

- **Best Model**: **Random Forest Classifier** was selected for production since it achieved the highest F1-Score (**82.41%**) and ROC-AUC (**0.7913**).
- Saved binary targets are stored in the `model/` folder.

---

## IBM Watson ML Cloud Deployment

The application is fully prepared for remote deployment.

### Deployment Configuration Steps:
1. **Save and package model assets**:
   Zip `model/model.pkl`, `model/scaler.pkl`, and `model/encoder.pkl`.

2. **Upload to Cloud Object Storage (COS)**:
   Upload the zip archive as a dataset asset inside your IBM Cloud account.

3. **IBM Watson Machine Learning Setup**:
   - Create a Watson Machine Learning Service instance inside your IBM Cloud Catalog.
   - Create a Deployment Space, link it to your Object Storage, and import the model asset.
   - Deploy the model asset as an **Online Deployment**.

4. **Integration**:
   Swap your credentials (API Key, Space ID, Deployment ID, and Cloud Region URL) into the class variables of `wml_deploy.py`. The Flask app will communicate with the online WML deployment via standard HTTPS requests.

---

## Running the Application

1. **Initialize the Dataset and Preprocess**:
   ```bash
   python generate_dataset.py
   python notebooks/preprocessing_eda.py
   ```

2. **Train and Select Model**:
   ```bash
   python train_model.py
   ```

3. **Start the Web App**:
   ```bash
   python app.py
   ```

4. **Access the Portal**:
   Open a browser and navigate to `http://127.0.0.1:5000/`.
   - Sign up a new analyst account via `Register`.
   - Log in to view the active model metrics, recent decisions, and EDA charts.
   - Enter applicant data to execute real-time underwriting scoring!

---

## Future Scope & Conclusion

### Future Scope
- **Real-Time Bureau Integration**: Connect the database layer to API integrations retrieving credit records directly from credit bureaus.
- **Explainable AI (XAI)**: Integrate libraries like SHAP or LIME to explain *why* the model approved or rejected a card application.
- **Deep Learning Classifiers**: Compare model performance with Multi-Layer Perceptron (MLP) networks.

### Conclusion
This project successfully delivers a professional, secure, and production-ready Credit Card Approval Prediction application. By decoupling layers (Presentation, Application, ML, DB, and Cloud Deployment) and training multiple machine learning algorithms, the platform offers an industry-standard solution for automated credit underwriting.
