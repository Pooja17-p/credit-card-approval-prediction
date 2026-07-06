import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

def train_and_evaluate():
    print("-----------------------------------------")
    print("PHASE 3: MACHINE LEARNING MODEL TRAINING")
    print("-----------------------------------------")

    os.makedirs('model', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)

    # 1. Load split datasets
    X_train = pd.read_csv('dataset/X_train.csv')
    X_test = pd.read_csv('dataset/X_test.csv')
    # Read y as Series
    y_train = pd.read_csv('dataset/y_train.csv').iloc[:, 0]
    y_test = pd.read_csv('dataset/y_test.csv').iloc[:, 0]

    # Define columns
    num_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Age', 'Years_Employed', 'Income_Per_Member']
    cat_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
    bin_cols = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'Is_Retired']

    # 2. Fit Scaler and Encoder separately
    print("\n[Task 14] Fitting Scaler and Encoder...")
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[num_cols])
    X_test_num_scaled = scaler.transform(X_test[num_cols])

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat_encoded = encoder.fit_transform(X_train[cat_cols])
    X_test_cat_encoded = encoder.transform(X_test[cat_cols])

    # Convert binary columns to numpy arrays
    X_train_bin = X_train[bin_cols].values
    X_test_bin = X_test[bin_cols].values

    # Concatenate features: [Scaled Numerical | Encoded Categorical | Pass-through Binary]
    X_train_proc = np.hstack([X_train_num_scaled, X_train_cat_encoded, X_train_bin])
    X_test_proc = np.hstack([X_test_num_scaled, X_test_cat_encoded, X_test_bin])

    # Save preprocessing objects
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(encoder, 'model/encoder.pkl')
    print("Saved model/scaler.pkl and model/encoder.pkl")

    # Keep track of feature list structure
    cat_features = list(encoder.get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_features + bin_cols
    print(f"Total processed features: {len(feature_names)}")

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=6, eval_metric='logloss', random_state=42)
    }

    results = []

    # Train and evaluate each model
    for name, clf in models.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_proc, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test_proc)
        y_prob = clf.predict_proba(X_test_proc)[:, 1] if hasattr(clf, "predict_proba") else [0]*len(y_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        
        print(f"{name} Evaluation:")
        print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
        
        # Save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Rejected (0)', 'Approved (1)'], 
                    yticklabels=['Rejected (0)', 'Approved (1)'])
        plt.title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        filename_cm = f"static/images/cm_{name.lower().replace(' ', '_')}.png"
        plt.savefig(filename_cm, dpi=300)
        plt.close()

    # Compare and print results
    results_df = pd.DataFrame(results)
    print("\n=======================================================")
    print("MODEL PERFORMANCE COMPARISON")
    print("=======================================================")
    print(results_df.to_string(index=False))
    print("=======================================================")

    # Select the best model based on F1-Score
    best_row = results_df.loc[results_df['F1-Score'].idxmax()]
    best_model_name = best_row['Model']
    print(f"\nBest Model Selected based on F1-Score: {best_model_name}")

    # Save the best model
    best_clf = models[best_model_name]
    joblib.dump(best_clf, 'model/model.pkl')
    print("Saved best model to model/model.pkl")

    # Plot model comparison bar chart
    plt.figure(figsize=(10, 6))
    melted_results = pd.melt(results_df, id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], 
                             var_name='Metric', value_name='Score')
    sns.barplot(data=melted_results, x='Model', y='Score', hue='Metric', palette='muted')
    plt.ylim(0, 1.05)
    plt.title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('static/images/model_comparison.png', dpi=300)
    plt.close()
    print("Saved comparison chart to static/images/model_comparison.png")

if __name__ == "__main__":
    train_and_evaluate()
