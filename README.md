Heart Disease Predictor
Overview
This is a beginner-level machine learning project to predict heart disease using the UCI Heart Disease dataset from Kaggle. It uses a Random Forest Classifier to predict whether a person has heart disease based on features like age, blood pressure, cholesterol, and more. The project was developed in a Jupyter Notebook (renamed to heart_disease_predictor.ipynb from Untitled1.ipynb).
Dataset

Source: Kaggle UCI Heart Disease dataset
Details: 920 rows, 16 columns (e.g., age, sex, chest pain type, cholesterol, target: num)
Preprocessing:
Filled missing numerical values with mean, categorical with 'Unknown'
One-hot encoded categorical features
Scaled numerical features using StandardScaler
Split data into 80% training, 20% testing



Technologies Used

Language: Python 3 (Google Colab)
Libraries:
Pandas: Data manipulation
Scikit-learn: RandomForestClassifier, StandardScaler, train_test_split, metrics
Joblib: Save/load model and scaler


Methods:
Random Forest Classifier (100 trees)
One-hot encoding for categorical data
StandardScaler for numerical data
Model evaluation with accuracy, classification report, confusion matrix



Workflow

Setup: Uploaded Kaggle API key, downloaded dataset via Kaggle API
Data Prep: Loaded data, handled missing values, encoded features, scaled data
Training: Trained Random Forest Classifier
Evaluation: Achieved 95% accuracy on test data
Saving: Saved model (heart_rf_model.pkl) and scaler (heart_scaler.pkl)
Prediction: Created a template CSV (Heart_user_template.csv) for user inputs; predicts heart disease on new data

Results

Accuracy: 95% on test set
Predictions: Outputs 0 (no disease) or 1+ (disease) for user inputs
Strengths: Simple, high accuracy, beginner-friendly
Limitations: No hyperparameter tuning or cross-validation

How to Run

Clone the repo: git clone https://github.com/yourusername/heart-disease-predictor.git
Open heart_disease_predictor.ipynb in Jupyter or Google Colab
Install dependencies: !pip install kaggle pandas scikit-learn joblib
Upload Kaggle API key (kaggle.json) when prompted
Run cells to train or predict using Heart_user_template.csv

Files

heart_disease_predictor.ipynb: Main notebook
Heart_user_template.csv: Template for user input
heart_rf_model.pkl: Trained model
heart_scaler.pkl: Scaler for preprocessing

Author: Manya K N Date: August 29, 2025
