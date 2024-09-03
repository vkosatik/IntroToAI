import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#first need to convert the data appropriately. Encode action string to number values and rewrite \
# the booleans as booleans
data = pd.read_excel('set2_500_patients.xlsx', engine='openpyxl')
data = data[["day", "hour", "action", "alertness", "hypertension", "intoxication", "heart_attack", "accident", "smoker", "overweight", "family_history", "goof_ball", "subject"]]
predict = "heart_attack"
data['heart_attack'] = data['heart_attack'].astype(bool)
data['accident'] = data['accident'].astype(bool)
string_column = ["action"]
encoder = LabelEncoder()
data['action'] = encoder.fit_transform(data['action'])
scaler = MinMaxScaler()
data['action'] = scaler.fit_transform(data[['action']])

#let's predict heart attacks and death
class Patient:
    def __init__(self, day, hour, action, alertness, hypertension, intoxication, smoker, overweight, family_history, goof_ball, subject):
        self.day = day
        self.hour = hour
        self.action = action
        self.alertness = alertness
        self.hypertension = hypertension
        self.intoxication = intoxication
        self.smoker = smoker
        self.overweight = overweight
        self.family_history = family_history
        self.goof_ball = goof_ball
        self.subject = subject

patientList = []
for i in range(len(data)):
    patientList.append(Patient(data.iloc[i,0], data.iloc[i,1], data.iloc[i,2], data.iloc[i,3], data.iloc[i,4], data.iloc[i,5], data.iloc[i,8], data.iloc[i,9], data.iloc[i,10], data.iloc[i,11], data.iloc[i,12]))


    def predict_heart_attack(self):
        if(self.hypertension > 0.5):
            if(self.intoxication > 0.5):
                return True
            else:
                return False

    def predict_accident(self):
        if(self.alertness < 0.5):
            if(self.intoxication > 0.5):
                return True
            else:
                return False

    def compare_ha_prediction(self):
        if(self.heart_attack == self.predict_heart_attack()):
            return True
        else:
            return False
    def compare_acc_prediction(self):
        if(self.accident == self.predict_accident()):
            return True
        else:
            return False
    def predict_all_patients(self):
        if(self.compare_ha_prediction() and self.compare_acc_prediction()):
            return True
        else:
            return False

def main():
    correct = 0
    for i in range(len(patientList)):
        if(patientList[i].predict_all_patients()):
            correct += 1
    print(correct/len(patientList))

main()