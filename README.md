# Churn-Prediction-Using-ML

**🧠 Customer Churn Prediction Project
📝 Project Description**

This project aims to predict whether a telecom customer is likely to discontinue their service (churn) based on demographic, account, and service usage details. Using machine learning techniques, the model identifies potential churners, helping the company take preventive measures such as offering personalized discounts or better service plans.
The project includes the full pipeline — from data preprocessing and transformation to model training, evaluation, and deployment using Flask and Python.

**📂 Dataset Overview**
**Dataset Name:** Telco Customer Churn
**Total Records:** 7043
**Total Attributes:** 20 input features + 1 target variable
**Target Variable:** Churn (Yes → churned, No → stayed)

**🔑 Feature Highlights**
Feature	Description
gender, SeniorCitizen	Basic customer demographics
Partner, Dependents	Relationship information
PhoneService, MultipleLines	Telecommunication service details
InternetService, OnlineSecurity, OnlineBackup	Internet plan and add-ons
Contract, PaymentMethod	Customer contract and billing method
tenure, MonthlyCharges, TotalCharges	Numerical features showing customer engagement and payments

**⚙ Workflow Summary**

**1️⃣ Data Preprocessing**
Handled missing values and inconsistent entries in the TotalCharges column.
Ensured all categorical and numerical data types were consistent.
Applied log transformation to reduce skewness in numerical columns (tenure, TotalCharges).

**2️⃣ Feature Engineering**
Added derived columns like tenure_log_trim and TotalCharges_log_trim to improve model interpretability.
Removed redundant or low-impact features using feature importance and correlation analysis.

**3️⃣ Encoding & Transformation**
One-Hot Encoding: Applied to categorical columns such as InternetService, PaymentMethod, etc.
Scaling: Used StandardScaler to normalize numerical features for balanced model performance.
Label Encoding: Converted Churn variable into binary format (Yes → 1, No → 0).

**4️⃣ Balancing the Dataset**
Implemented SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.
Improved recall for minority class (Churn = Yes) by generating synthetic samples.

**5️⃣ Model Training**
Multiple models were trained and compared, including:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Random Forest Classifier
Support Vector Machine (SVM)
Gradient Boosting Classifier (final selected model)

**6️⃣ Evaluation Metrics**
Accuracy
Precision
Recall
F1-Score
ROC–AUC Curve
Final Model: Gradient Boosting Classifier
Accuracy: 79%
ROC–AUC: 0.81
Balanced Performance: Optimized precision and recall post-SMOTE.

**🌐 Deployment**
**Backend Framework:** Flask
**Frontend:** HTML + CSS (index.html interface)
**Hosting Platform:** Render
**Model Files Used:**
            churn_prediction_project.pkl (Trained Model)
            standard_scaler.pkl (Scaler for normalization)
            feature_columns.pkl (Preserved feature list)
Users can input customer information through the web interface. The backend processes inputs, scales data, applies encoding, and displays whether the customer is likely to churn or stay.

**🧰 Tools & Technologies**
**Programming Language:** Python
**Libraries:** Pandas, NumPy, scikit-learn, imblearn, xgboost
**Visualization:** Matplotlib, Seaborn
**Deployment Tools:** Flask, Render

**📈 Key Insights**
Customers with short-term contracts tend to churn more frequently.
High monthly charges are linked with increased churn probability.
Electronic check payments show higher churn rates compared to automatic payments.
Long-term tenure and fiber optic users significantly influence retention.
Customers using multiple online services (security, backup) are less likely to leave.

**👨‍💻 Support**
**Developer:** Thalari Nagasuresh
**Focus Area:** Machine Learning & Predictive Analytics
**Deployment:** Render – Customer Churn Prediction Web App
