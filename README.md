# Bank_Dataset
This repository contains Python code for predicting bank defaults based on various customer features. The code uses a dataset (bank_data_g18.csv) containing information about customers, such as income, age, loan amount, overdue days, marital status, house ownership, car ownership, city, and product, among others. The code employs logistic regression for the prediction task.
# Code Overview
bank_default_prediction.py: The main Python script containing the code for data preprocessing, feature engineering, model training, and evaluation. It uses logistic regression for binary classification to predict bank defaults based on selected features.
# Data Preprocessing
Handling Missing Values: The script checks for missing values in the dataset and handles them appropriately.
Feature Selection: The script drops irrelevant features (Profession, CIF_Id, etc.) and selects relevant features for the prediction task.
Outlier Detection and Removal: Outliers are detected using boxplots and removed from the dataset.
Weight of Evidence (WoE) Transformation: Categorical features (Married/Single, House_Ownership, Car_Ownership, City, Product) are transformed using WoE to capture their relationship with the target variable.
# Model Training and Evaluation
Feature Scaling: No feature scaling is required for logistic regression.
Model Training: The logistic regression model is trained using the preprocessed features.
Model Evaluation: The script evaluates the model's performance using metrics such as Gini coefficient, ROC-AUC curve, confusion matrix, and classification report on the test dataset. The evaluation is performed for both the balanced and unbalanced datasets.
# Results
The script outputs the evaluation metrics for both the balanced and unbalanced datasets. Additionally, it provides ROC-AUC curves and Gini coefficients for visual representation of the model's performance.
# Conclusion
The code demonstrates a predictive modeling approach for bank default prediction using logistic regression. 
