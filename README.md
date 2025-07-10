# Diabetes Prediction Model

This project uses a machine learning model to predict the likelihood of diabetes in a patient based on diagnostic health parameters. It leverages the Pima Indians Diabetes dataset and is built using Python with scikit-learn.

## Overview

The goal of this project is to apply supervised learning techniques to develop a binary classification model that can determine whether an individual is diabetic or not. It includes data preprocessing, model training using Support Vector Machine (SVM), and evaluation on both training and test data.

## Dataset

The dataset used is the **Pima Indians Diabetes Dataset**, which contains medical information for female patients of Pima Indian heritage. The dataset includes the following features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0: Non-diabetic, 1: Diabetic)

## Technologies Used

- Python
- pandas
- numpy
- scikit-learn (SVM, preprocessing, model selection, metrics)

## Workflow

1. **Data Preprocessing**
   - Loaded dataset with pandas
   - Split features (`X`) and target (`Y`)
   - Standardized features using `StandardScaler`

2. **Model Training**
   - Split data into training and test sets (80:20 split)
   - Used SVM with a linear kernel for classification

3. **Model Evaluation**
   - Accuracy evaluated on both training and test sets using `accuracy_score`

4. **Prediction System**
   - Custom input data is used to simulate predictions for new patient data

## Example Prediction

Given an input tuple like:
(5, 166, 72, 19, 175, 25.8, 0.587, 51)


The model outputs whether the person is likely diabetic or not based on trained patterns.

## How to Run

1. Clone the repository
2. Make sure `diabetes.csv` is in the same directory
3. Run the script:

```bash
python diabetes_prediction.py

----
