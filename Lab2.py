import numpy as np
import pandas as pd
import os
import pickle
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, \
    mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

import statsmodels.api as sm


#test different ML models for
# predicting accident and heart attack
# predicting activity class
# predicting continuous measure
# forecasting activity for (all dimensions for the next hour based on the past 8 hours for example)

# !!! don't have the same subject in training test and validation but divide it by subject
# don't forget metrics
#report the results

def process_and_combine_datasets():
    dataset1 = pd.read_excel('set2_500_patients.xlsx', engine='openpyxl')
    dataset1 = dataset1.iloc[:, 1:]
    dataset2 = pd.read_excel('set3_500_patients.xlsx', engine='openpyxl')
    dataset2 = dataset2.iloc[:, 1:]
    combined_data = pd.concat([dataset1, dataset2], axis=0)
    return combined_data

pickle_path = 'combined_data.pkl'
subjects_pickle_path = 'shuffled_subjects.pkl'

if os.path.exists(pickle_path):
    with open(pickle_path, 'rb') as f:
        combined_data = pickle.load(f)
else:
    combined_data = process_and_combine_datasets()
    with open(pickle_path, 'wb') as f:
        pickle.dump(combined_data, f)

combined_data['heart_attack'] = combined_data['heart_attack'].astype(bool)
combined_data['accident'] = combined_data['accident'].astype(bool)
grouped = combined_data.groupby('subject')

if os.path.exists(subjects_pickle_path):
    with open(subjects_pickle_path, 'rb') as f:
        subjects = pickle.load(f)
else:
    subjects = list(grouped.groups.keys())
    np.random.shuffle(subjects)
    with open(subjects_pickle_path, 'wb') as f:
        pickle.dump(subjects, f)

num_subjects = len(subjects)
train_end = int(0.7 * num_subjects)
val_end = int(0.85 * num_subjects)

train_subjects = subjects[:train_end]
val_subjects = subjects[train_end:val_end]
test_subjects = subjects[val_end:]

train_data = pd.concat([grouped.get_group(subj) for subj in train_subjects])
val_data = pd.concat([grouped.get_group(subj) for subj in val_subjects])
test_data = pd.concat([grouped.get_group(subj) for subj in test_subjects])

#let's create a model that predicts the activity class now
#don't forget cross validation from scikit

# X_train = train_data.drop(columns=['action', 'heart_attack', 'accident', 'subject'])
# y_train = train_data['action']

# X_val = val_data.drop(columns=['action', 'heart_attack', 'accident', 'subject'])
# y_val = val_data['action']

# X_test = test_data.drop(columns=['action', 'heart_attack', 'accident', 'subject'])
# y_test = test_data['action']

# # Initialize the Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)

# # Train the model
# rf_classifier.fit(X_train, y_train)

# # Validate the model
# y_val_pred = rf_classifier.predict(X_val)
# val_accuracy = accuracy_score(y_val, y_val_pred)
# print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
# print('Validation Classification Report:')
# print(classification_report(y_val, y_val_pred, zero_division=0))

# # Test the model
# y_test_pred = rf_classifier.predict(X_test)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
# print('Test Classification Report:')
# print(classification_report(y_test, y_test_pred, zero_division=0))

# # Confusion Matrix for Test Data
# conf_matrix = confusion_matrix(y_test, y_test_pred)
# print('Confusion Matrix for Test Data:')
# print(conf_matrix)

# # Plot the confusion matrix
# labels = ['alcohol', 'coffee', 'nothing', 'patient died', 'sleep', 'work']
# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# scores = cross_val_score(rf_classifier, X_train, y_train, cv=5, scoring='accuracy')
# print(f'Cross-validation scores: {scores}')
# print(f'Mean cross-validation score: {scores.mean():.2f}')

# Uncommented the regression part
X_train_intoxication = train_data.drop(columns=['intoxication', 'action', 'heart_attack', 'accident', 'subject'])
y_train_intoxication = train_data['intoxication']

X_val_intoxication = val_data.drop(columns=['intoxication', 'action', 'heart_attack', 'accident', 'subject'])
y_val_intoxication = val_data['intoxication']

X_test_intoxication = test_data.drop(columns=['intoxication', 'action', 'heart_attack', 'accident', 'subject'])
y_test_intoxication = test_data['intoxication']

# Initialize the Linear Regression model
linear_regressor = LinearRegression()

# Train the model
linear_regressor.fit(X_train_intoxication, y_train_intoxication)

# Validate the model
y_val_pred_intoxication = linear_regressor.predict(X_val_intoxication)
val_mae = mean_absolute_error(y_val_intoxication, y_val_pred_intoxication)
val_mse = mean_squared_error(y_val_intoxication, y_val_pred_intoxication)
val_r2 = r2_score(y_val_intoxication, y_val_pred_intoxication)
print(f'Validation MAE: {val_mae:.2f}')
print(f'Validation MSE: {val_mse:.2f}')
print(f'Validation R²: {val_r2:.2f}')

# Test the model
y_test_pred_intoxication = linear_regressor.predict(X_test_intoxication)
test_mae = mean_absolute_error(y_test_intoxication, y_test_pred_intoxication)
test_mse = mean_squared_error(y_test_intoxication, y_test_pred_intoxication)
test_r2 = r2_score(y_test_intoxication, y_test_pred_intoxication)
print(f'Test MAE: {test_mae:.2f}')
print(f'Test MSE: {test_mse:.2f}')
print(f'Test R²: {test_r2:.2f}')

X_train_intoxication_sm = sm.add_constant(X_train_intoxication)  # Add a constant term for the intercept
model = sm.OLS(y_train_intoxication, X_train_intoxication_sm).fit()
print(model.summary())