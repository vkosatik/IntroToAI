import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
data = data.iloc[:,3:]
data = data.iloc[:,:-1]

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
logistic = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
logistic.fit(x_train, y_train)
acc = logistic.score(x_test, y_test)
print(data)
print(str(acc*100)+'%')

#need to check for if the heart attack is true or accident then what is the accuracy there
# since there are majority of false so it could be wrongfully high accuracy


