import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('diabetes.csv')

# Separating features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardizing data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Splitting into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Model training
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Model evaluation
X_train_prediction = classifier.predict(X_train)
training_accuracy = accuracy_score(Y_train, X_train_prediction)

X_test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Training Accuracy:", training_accuracy)
print("Test Accuracy:", test_accuracy)


input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

input_df = pd.DataFrame([input_data], columns=[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
])

std_data = scaler.transform(input_df)
prediction = classifier.predict(std_data)

print("Prediction:", prediction)

if prediction[0] == 0:
    print("The person is NOT diabetic.")
else:
    print("The person IS diabetic.")
