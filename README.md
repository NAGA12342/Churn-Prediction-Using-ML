# 🧠 Customer Churn Prediction Using Machine Learning

## 📝 Project Description
This project aims to predict whether a telecom customer is likely to discontinue their service (**churn**) based on demographic, account, and service usage details.  
Using machine learning techniques, the model identifies potential churners, helping the company take preventive measures such as offering personalized discounts or better service plans.  

The project covers the **complete end-to-end pipeline** — from data preprocessing and feature engineering to model training, evaluation, and deployment using **Flask** and **Python**.

---

## 📂 Dataset Overview
- **Dataset Name:** WA_Fn-UseC_-Telco-Customer-Churn 
- **Total Records:** 7043  
- **Total Attributes:** 20 input features + 1 target variable  
- **Target Variable:** `Churn` (Yes → churned, No → stayed)

### 🔑 Feature Highlights
| Feature | Description |
|----------|--------------|
| `gender`, `SeniorCitizen` | Basic customer demographics |
| `Partner`, `Dependents` | Relationship information |
| `PhoneService`, `MultipleLines` | Telecommunication service details |
| `InternetService`, `OnlineSecurity`, `OnlineBackup` | Internet plan and add-ons |
| `Contract`, `PaymentMethod` | Customer contract and billing method |
| `tenure`, `MonthlyCharges`, `TotalCharges` | Numeric features showing customer engagement and payments |

---

## ⚙ Workflow Summary

### 1️⃣ Data Preprocessing
- Handled missing values and inconsistent entries in the **TotalCharges** column.  
- Ensured all categorical and numerical data types were consistent.  
- Applied log transformation to reduce skewness in numerical columns (`tenure`, `TotalCharges`).  

### 2️⃣ Feature Engineering
- Created derived columns such as `tenure_log_trim` and `TotalCharges_log_trim` for better interpretability.  
- Removed redundant or low-impact features using **feature importance** and **correlation analysis**.  

### 3️⃣ Encoding & Transformation
- **One-Hot Encoding:** Applied to categorical features (`InternetService`, `PaymentMethod`, etc.)  
- **Scaling:** Used `StandardScaler` to normalize numeric features for balanced performance.  
- **Label Encoding:** Converted `Churn` into binary format (Yes → 1, No → 0).  

### 4️⃣ Balancing the Dataset
- Implemented **SMOTE (Synthetic Minority Oversampling Technique)** to handle class imbalance.  
- Improved recall for the minority class (Churn = Yes) by generating synthetic samples.  

### 5️⃣ Model Training
Models trained and compared include:
- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- **Gradient Boosting Classifier (Final Model)**  

### 6️⃣ Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC–AUC Curve  

**Final Model:** Gradient Boosting Classifier  
- **Accuracy:** 79%  
- **ROC–AUC:** 0.81  
- **Result:** Balanced precision and recall after SMOTE  

---

## 🌐 Deployment
- **Backend Framework:** Flask  
- **Frontend:** HTML + CSS (`index.html` web interface)  
- **Hosting Platform:** Render  
- **Model Files Used:**
  - `churn_prediction_project.pkl` → Trained model  
  - `standard_scaler.pkl` → Scaler for normalization  
  - `feature_columns.pkl` → List of input feature columns  

Users can input customer information through the web interface.  
The backend processes the input, scales and encodes features, and predicts whether the customer is **likely to churn or stay**.

---

## 🧰 Tools & Technologies
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, scikit-learn, imblearn, XGBoost  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Flask, Render  

---

## 📈 Key Insights
- Customers with **short-term contracts** tend to churn more frequently.  
- **High monthly charges** are strongly linked to churn.  
- **Electronic check payments** have a higher churn rate compared to automatic payments.  
- **Longer tenure** and **fiber optic plans** increase customer retention.  
- Customers using **multiple value-added online services** (like security & backup) are less likely to leave.  

---

## 👨‍💻 Support
**Developer:** Thalari Nagasuresh  
**Focus Area:** Machine Learning & Predictive Analytics  
**Deployment:** https://churn-prediction-using-ml-fdbl.onrender.com/
