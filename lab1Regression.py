import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def process_excel_data(file_path):
    data = pd.read_excel(file_path, engine='openpyxl')
    data = data[
        ["day", "hour", "action", "alertness", "hypertension", "intoxication", "heart_attack", "accident", "smoker",
         "overweight", "family_history", "goof_ball", "subject"]]

    data['heart_attack'] = data['heart_attack'].astype(bool)
    data['accident'] = data['accident'].astype(bool)

    encoder = LabelEncoder()
    data['action'] = encoder.fit_transform(data['action'])
    scaler = MinMaxScaler()
    data['action'] = scaler.fit_transform(data[['action']])

    # Drop the first 3 columns and the last column
    data = data.iloc[:, 3:]
    data = data.iloc[:, :-1]

    return data

train_data = process_excel_data('set2_500_patients.xlsx')
test_data = process_excel_data('set3_500_patients.xlsx')

predict_ha = 'heart_attack'
predict_acc = 'accident'

x_train_ha = np.array(train_data.drop([predict_ha], axis=1))
y_train_ha = np.array(train_data[predict_ha])

x_train_acc = np.array(train_data.drop([predict_acc], axis=1))
y_train_acc = np.array(train_data[predict_acc])

x_test_ha = np.array(test_data.drop([predict_ha], axis=1))
y_test_ha = np.array(test_data[predict_ha])

x_test_acc = np.array(test_data.drop([predict_acc], axis=1))
y_test_acc = np.array(test_data[predict_acc])

model_ha = LogisticRegression()
model_ha.fit(x_train_ha, y_train_ha)
y_pred_ha = model_ha.predict(x_test_ha)

acc_model = LogisticRegression()
acc_model.fit(x_train_acc, y_train_acc)
y_pred_acc = acc_model.predict(x_test_acc)

accuracy_ha = accuracy_score(y_test_ha, y_pred_ha)
print(f"Accuracy for Heart Attacks Prediction: {accuracy_ha}")

print("Confusion Matrix for Heart Attack Prediction:")
print(confusion_matrix(y_test_ha, y_pred_ha))

print("Classification Report for Heart Attack Prediction:")
print(classification_report(y_test_ha, y_pred_ha))

accuracy_acc = accuracy_score(y_test_acc, y_pred_acc)
print(f"Accuracy for Accident Prediction: {accuracy_acc}")

print("Confusion Matrix for Accident Prediction:")
print(confusion_matrix(y_test_acc, y_pred_acc))

print("Classification Report for Accident Prediction:")
print(classification_report(y_test_acc, y_pred_acc))


# Accuracy for positive heart attack cases
positive_ha_indices = np.where(y_test_ha == True)[0]
x_test_ha_positive = x_test_ha[positive_ha_indices]
y_test_ha_positive = y_test_ha[positive_ha_indices]
y_pred_ha_positive = model_ha.predict(x_test_ha_positive)

accuracy_ha_positive = accuracy_score(y_test_ha_positive, y_pred_ha_positive)
print(f"Accuracy for positive only Heart Attack Cases: {accuracy_ha_positive}")

#print("Confusion Matrix for Positive Heart Attack Cases:")
#print(confusion_matrix(y_test_ha_positive, y_pred_ha_positive))

#print("Classification Report for Positive Heart Attack Cases:")
#print(classification_report(y_test_ha_positive, y_pred_ha_positive))

# Accuracy for positive accident cases
positive_acc_indices = np.where(y_test_acc == True)[0]
x_test_acc_positive = x_test_acc[positive_acc_indices]
y_test_acc_positive = y_test_acc[positive_acc_indices]
y_pred_acc_positive = acc_model.predict(x_test_acc_positive)

accuracy_acc_positive = accuracy_score(y_test_acc_positive, y_pred_acc_positive)
print(f"Accuracy for positive only Accident Cases: {accuracy_acc_positive}")

#print("Confusion Matrix for Positive Accident Cases:")
#print(confusion_matrix(y_test_acc_positive, y_pred_acc_positive))

#print("Classification Report for Positive Accident Cases:")
#rint(classification_report(y_test_acc_positive, y_pred_acc_positive))