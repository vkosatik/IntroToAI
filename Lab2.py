import numpy as np
import pandas as pd
import os
import pickle


#test different ML models for
# predicting accident and heart attack
# predicting activity class
# predicting continuous measure
# forecasting activity for (all dimensions for the next hour based on the past 8 hours for example)

# !!! don't have the same subject in training test and validation but divide it by subject
# don't forget metrics
#report

def process_and_combine_datasets():
    dataset1 = pd.read_excel('set2_500_patients.xlsx', engine='openpyxl')
    dataset1 = dataset1.iloc[:, 1:]
    dataset2 = pd.read_excel('set3_500_patients.xlsx', engine='openpyxl')
    dataset2 = dataset2.iloc[:, 1:]
    combined_data = pd.concat([dataset1,dataset2], axis=0)
    return combined_data

pickle_path = 'combined_data.pkl'

if os.path.exists(pickle_path):
    with open(pickle_path, 'rb') as f:
        combined_data = pickle.load(f)
else:
    combined_data = process_and_combine_datasets()
    with open(pickle_path, 'wb') as f:
        pickle.dump(combined_data,f)

combined_data['heart_attack'] = combined_data['heart_attack'].astype(bool)
combined_data['accident'] = combined_data['accident'].astype(bool)
grouped = combined_data.groupby('subject')

subjects = grouped.groups.keys()
subjects = list(subjects)
#np.random.shuffle(subjects)

num_subjects = len(subjects)
train_end = int(0.7 * num_subjects)
val_end = int(0.85 * num_subjects)

train_subjects = subjects[:train_end]
val_subjects = subjects[train_end:val_end]
test_subjects = subjects[val_end:]

train_data = pd.concat([grouped.get_group(subj) for subj in train_subjects])
val_data = pd.concat([grouped.get_group(subj) for subj in val_subjects])
test_data = pd.concat([grouped.get_group(subj) for subj in test_subjects])

print('train data')
print(train_data)
print('val data')
print(val_data)
print('test data')
print(test_data)


print('Training size:', len(train_data))
print('Validation size', len(val_data))
print('Test size', len(test_data))

counter1_ha = train_data['heart_attack'].sum()
counter2_ha = val_data['heart_attack'].sum()
counter3_ha = test_data['heart_attack'].sum()
counter1_acc = train_data['accident'].sum()
counter2_acc = val_data['accident'].sum()
counter3_acc = test_data['accident'].sum()

print('Heart attack distribution:')
print(counter1_ha, counter2_ha, counter3_ha)

print('Accident distribution:')
print(counter1_acc, counter2_acc, counter3_acc)

print(841/134327)
print(166/(238031-134328))
print(188/(341074-238032))

print((841+166+188)/341074)