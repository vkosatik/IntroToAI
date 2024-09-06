import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix


def rule_process_excel_data(file_path):
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


    return data

processed_test_data = rule_process_excel_data('set3_500_patients.xlsx')
processed_test_data['history_of_heart_attack'] = 0

subject_history = {}
for index, row in processed_test_data.iterrows():
    subject_id = row['subject']
    if subject_id not in subject_history:
        subject_history[subject_id] = 0
    if row['heart_attack']:
        subject_history[subject_id] += 1
    processed_test_data.at[index, 'history_of_heart_attack'] = subject_history[subject_id]




def rule_predict_ha(subject):
    score = 0

    if subject['hypertension']> 0.4:
        score += 0.4
    if subject['smoker']>0.3:
        score += 0.3

    if subject['overweight']>0.2:
        score += 0.2
    if subject['family_history']>0.1:
        score += 0.1
    if subject['history_of_heart_attack']>0:
        score += 0.1 * subject['history_of_heart_attack']
    if score>0.6:
        return True
    return False

def rule_predict_acc(subject):
    score = 0
    if subject['goof_ball'] >0.3:
        score += 0.3
    if subject['intoxication'] > 0.3:
        score += 0.3
    if subject['alertness']>0.2:
        score += 0.2
    if subject['overweight']>0.2:
        score += 0.2
    if score>0.4:
        return True
    return False

#now add new columns with the preditions
predicted_ha = []
predicted_acc = []

for index, row in processed_test_data.iterrows():
    predicted_ha.append(rule_predict_ha(row))
    predicted_acc.append(rule_predict_acc(row))

processed_test_data['predicted_ha'] = predicted_ha
processed_test_data['predicted_acc'] = predicted_acc

correct_ha = 0
correct_acc = 0
#compare
for index, row in processed_test_data.iterrows():
    if row['predicted_ha'] == row['heart_attack']:
        correct_ha += 1
    if row['predicted_acc'] == row['accident']:
        correct_acc += 1

rule_accuracy_ha = correct_ha / len(processed_test_data)
rule_accuracy_acc = correct_acc / len(processed_test_data)

print('Rule-based Accuracy for Heart Attack Prediction:  '+ str(rule_accuracy_ha*100) + '%')
print('Rule-based Accuracy for Accident Prediction: ' + str(rule_accuracy_acc*100) + '%')

ha_conf_matrix = confusion_matrix(processed_test_data['heart_attack'], processed_test_data['predicted_ha'])
ha_false_negatives = ha_conf_matrix[1, 0]

# Calculate confusion matrix for accident predictions
acc_conf_matrix = confusion_matrix(processed_test_data['accident'], processed_test_data['predicted_acc'])
acc_false_negatives = acc_conf_matrix[1, 0]
print('Confusion Matrix for Heart Attack Prediction:')
print(ha_conf_matrix)
print('False Negatives for Heart Attack Prediction:', ha_false_negatives)

print('Confusion Matrix for Accident Prediction:')
print(acc_conf_matrix)
print('False Negatives for Accident Prediction:', acc_false_negatives)