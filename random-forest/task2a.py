# Data Processing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import seaborn as sns
import graphviz

DU_data = pd.read_csv('./csv_files/DU_task2a.csv')
DD_data = pd.read_csv('./csv_files/DD_task2a.csv')
UU_data = pd.read_csv('./csv_files/UU_task2a.csv')
UD_data = pd.read_csv('./csv_files/UD_task2a.csv')
trigraph_data = pd.read_csv('./csv_files/trigraph_task2a.csv')

accuracy_data = pd.read_csv('./csv_files/accuracy_task2a.csv')
speed_data = pd.read_csv('./csv_files/speed_task2a.csv')
reaction_data = pd.read_csv('./csv_files/reaction_task2a.csv')

# merged arrays


merged_DU_accuracy_data = pd.merge(DU_data, accuracy_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_DU_speed_data = pd.merge(DU_data, speed_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_DU_UU_data = pd.merge(DU_data, UU_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_DU_reaction_data = pd.merge(UD_data,reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 


merged_UD_accuracy_data = pd.merge(UD_data, accuracy_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_UD_speed_data = pd.merge(UD_data, speed_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_UD_UU_data = pd.merge(UD_data, UU_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_UD_reaction_data = pd.merge(UD_data, reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 
# print(DU_data.target_names)
# print(DU_data.feature_names)

DU_nopred = DU_data.drop(['user','session', 'task', 'iteration'], axis=1)
DU_pred= DU_data['user']

DD_nopred = DD_data.drop(['user','session', 'task', 'iteration'], axis=1)
DD_pred= DD_data['user']

UU_nopred = UU_data.drop(['user','session', 'task', 'iteration'], axis=1)
UU_pred= UU_data['user']

UD_nopred = UD_data.drop(['user','session', 'task', 'iteration'], axis=1)
UD_pred= UD_data['user']

trigraph_nopred = trigraph_data.drop(['user','session', 'task', 'iteration'], axis=1)
trigraph_pred= trigraph_data['user']

merged_DU_accuracy_data_nopred =  merged_DU_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DU_accuracy_data_pred= merged_DU_accuracy_data['user']

merged_DU_speed_data_nopred =  merged_DU_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DU_speed_data_pred= merged_DU_speed_data['user']

merged_DU_UU_data_nopred = merged_DU_UU_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DU_UU_data_pred= merged_DU_UU_data['user']

merged_DU_reaction_data_nopred = merged_DU_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DU_reaction_data_pred= merged_DU_reaction_data['user']

merged_UD_accuracy_data_nopred =  merged_UD_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UD_accuracy_data_pred= merged_UD_accuracy_data['user']

merged_UD_speed_data_nopred =  merged_UD_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UD_speed_data_pred= merged_UD_speed_data['user']

merged_UD_UU_data_nopred =  merged_UD_UU_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UD_UU_data_pred= merged_UD_UU_data['user']

merged_UD_reaction_data_nopred =  merged_UD_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UD_reaction_data_pred= merged_UD_reaction_data['user']

# Split the data into training and test sets
DU_nopred_train, DU_nopred_test, DU_pred_train, DU_pred_test = train_test_split(DU_nopred, DU_pred, shuffle=True, test_size=0.4)

DD_nopred_train, DD_nopred_test, DD_pred_train, DD_pred_test = train_test_split(DD_nopred, DD_pred, shuffle=True, test_size=0.4)

UU_nopred_train, UU_nopred_test, UU_pred_train, UU_pred_test = train_test_split(UU_nopred, UU_pred,  shuffle=True, test_size=0.4)

UD_nopred_train, UD_nopred_test, UD_pred_train, UD_pred_test = train_test_split(UD_nopred, UD_pred,shuffle=True, test_size=0.4)

trigraph_nopred_train, trigraph_nopred_test, trigraph_pred_train, trigraph_pred_test = train_test_split(trigraph_nopred, trigraph_pred, shuffle=True, test_size=0.4)

merged_DU_accuracy_data_nopred_train, merged_DU_accuracy_data_nopred_test, merged_DU_accuracy_data_pred_train, merged_DU_accuracy_data_pred_test = train_test_split(merged_DU_accuracy_data_nopred, merged_DU_accuracy_data_pred, shuffle=True, test_size=0.4)

merged_DU_speed_data_nopred_train, merged_DU_speed_data_nopred_test, merged_DU_speed_data_pred_train, merged_DU_speed_data_pred_test = train_test_split(merged_DU_speed_data_nopred, merged_DU_speed_data_pred, shuffle=True, test_size=0.4)

merged_DU_UU_data_nopred_train, merged_DU_UU_data_nopred_test, merged_DU_UU_data_pred_train, merged_DU_UU_data_pred_test = train_test_split(merged_DU_UU_data_nopred, merged_DU_UU_data_pred, shuffle=True, test_size=0.4)

merged_DU_reaction_data_nopred_train, merged_DU_reaction_data_nopred_test, merged_DU_reaction_data_pred_train, merged_DU_reaction_data_pred_test = train_test_split(merged_DU_reaction_data_nopred, merged_DU_reaction_data_pred, shuffle=True, test_size=0.4)

merged_UD_accuracy_data_nopred_train, merged_UD_accuracy_data_nopred_test, merged_UD_accuracy_data_pred_train, merged_UD_accuracy_data_pred_test = train_test_split(merged_UD_accuracy_data_nopred, merged_UD_accuracy_data_pred, shuffle=True, test_size=0.4)

merged_UD_speed_data_nopred_train, merged_UD_speed_data_nopred_test, merged_UD_speed_data_pred_train, merged_UD_speed_data_pred_test = train_test_split(merged_UD_speed_data_nopred, merged_UD_speed_data_pred, shuffle=True, test_size=0.4)

merged_UD_UU_data_nopred_train, merged_UD_UU_data_nopred_test, merged_UD_UU_data_pred_train, merged_UD_UU_data_pred_test = train_test_split(merged_UD_UU_data_nopred, merged_UD_UU_data_pred, shuffle=True, test_size=0.4)

merged_UD_reaction_data_nopred_train, merged_UD_reaction_data_nopred_test, merged_UD_reaction_data_pred_train, merged_UD_reaction_data_pred_test = train_test_split(merged_UD_reaction_data_nopred, merged_UD_reaction_data_pred, shuffle=True, test_size=0.4)



rf1 = RandomForestClassifier()
rf1.fit(DU_nopred_train, DU_pred_train)

rf2 = RandomForestClassifier()
rf2.fit(DD_nopred_train, DD_pred_train)

rf3 = RandomForestClassifier()
rf3.fit(UU_nopred_train, UU_pred_train)

rf4 = RandomForestClassifier()
rf4.fit(UD_nopred_train, UD_pred_train)

rf5 = RandomForestClassifier()
rf5.fit(trigraph_nopred_train, trigraph_pred_train)

rf6 = RandomForestClassifier()
rf6.fit(merged_DU_accuracy_data_nopred_train, merged_DU_accuracy_data_pred_train)

rf7 = RandomForestClassifier()
rf7.fit(merged_DU_speed_data_nopred_train, merged_DU_speed_data_pred_train)

rf8 = RandomForestClassifier()
rf8.fit(merged_DU_UU_data_nopred_train, merged_DU_UU_data_pred_train)

rf13 = RandomForestClassifier()
rf13.fit(merged_DU_reaction_data_nopred_train, merged_DU_reaction_data_pred_train)

rf9 = RandomForestClassifier()
rf9.fit(merged_UD_accuracy_data_nopred_train, merged_UD_accuracy_data_pred_train)

rf10 = RandomForestClassifier()
rf10.fit(merged_UD_speed_data_nopred_train, merged_UD_speed_data_pred_train)

rf11 = RandomForestClassifier()
rf11.fit(merged_UD_UU_data_nopred_train, merged_UD_UU_data_pred_train)

rf12 = RandomForestClassifier()
rf12.fit(merged_UD_reaction_data_nopred_train, merged_UD_reaction_data_pred_train)

# False negatives, False posives, ERR, etc
# make a curve

DU_prediction = rf1.predict(DU_nopred_test)
DU_accuracy = accuracy_score(DU_pred_test, DU_prediction)

DD_prediction = rf2.predict(DD_nopred_test)
DD_accuracy = accuracy_score(DD_pred_test, DD_prediction)

UU_prediction = rf3.predict(UU_nopred_test)
UU_accuracy = accuracy_score(UU_pred_test, UU_prediction)

UD_prediction = rf4.predict(UD_nopred_test)
UD_accuracy = accuracy_score(UD_pred_test, UD_prediction)

trigraph_prediction = rf5.predict(trigraph_nopred_test)
trigraph_accuracy = accuracy_score(trigraph_pred_test, trigraph_prediction)

merged_DU_accuracy_data_prediction = rf6.predict(merged_DU_accuracy_data_nopred_test)
merged_DU_accuracy_data_accuracy = accuracy_score(merged_DU_accuracy_data_pred_test, merged_DU_accuracy_data_prediction)

merged_DU_speed_data_prediction = rf7.predict(merged_DU_speed_data_nopred_test)
merged_DU_speed_data_accuracy = accuracy_score(merged_DU_speed_data_pred_test, merged_DU_speed_data_prediction)

merged_DU_UU_data_prediction = rf8.predict(merged_DU_UU_data_nopred_test)
merged_DU_UU_data_accuracy = accuracy_score(merged_DU_UU_data_pred_test, merged_DU_UU_data_prediction)

merged_DU_reaction_data_prediction = rf13.predict(merged_DU_reaction_data_nopred_test)
merged_DU_reaction_data_accuracy = accuracy_score(merged_DU_reaction_data_pred_test, merged_DU_reaction_data_prediction)

merged_UD_accuracy_data_prediction = rf9.predict(merged_UD_accuracy_data_nopred_test)
merged_UD_accuracy_data_accuracy = accuracy_score(merged_UD_accuracy_data_pred_test, merged_UD_accuracy_data_prediction)

merged_UD_speed_data_prediction = rf10.predict(merged_UD_speed_data_nopred_test)
merged_UD_speed_data_accuracy = accuracy_score(merged_UD_speed_data_pred_test, merged_UD_speed_data_prediction)

merged_UD_UU_data_prediction = rf11.predict(merged_UD_UU_data_nopred_test)
merged_UD_UU_data_accuracy = accuracy_score(merged_UD_UU_data_pred_test, merged_UD_UU_data_prediction)

merged_UD_reaction_data_prediction = rf12.predict(merged_UD_reaction_data_nopred_test)
merged_UD_reaction_data_accuracy = accuracy_score(merged_UD_reaction_data_pred_test, merged_UD_reaction_data_prediction)

# plt.figure(figsize=(16, 8))

# merged_UD_accuracy_data_corr = merged_UD_accuracy_data.corr()
# sns.heatmap(merged_UD_accuracy_data_corr, annot=True, square=True, fmt='0.2f')

# plt.show()

print("task 2a DU Accuracy:", DU_accuracy)
print("task 2a UU Accuracy:", UU_accuracy)
print("task 2a DD Accuracy:", DD_accuracy)
print("task 2a UD Accuracy:", UD_accuracy)
print("task 2a trigraph Accuracy:", trigraph_accuracy)

print("\n")

print("task 2a DU + typing accuracy Accuracy:", merged_DU_accuracy_data_accuracy)
print("task 2a DU + typing speed Accuracy:", merged_DU_speed_data_accuracy)
print("task 2a DU + UU Accuracy:", merged_DU_UU_data_accuracy)
print("task 2a DU + reaction time Accuracy:", merged_DU_reaction_data_accuracy)

print("\n")

print("task 2a UD + typing accuracy Accuracy:", merged_UD_accuracy_data_accuracy)
print("task 2a UD + typing speed Accuracy:", merged_UD_speed_data_accuracy)
print("task 2a UD + UU Accuracy:", merged_UD_UU_data_accuracy)
print("task 2a UD + reaction time Accuracy:", merged_UD_reaction_data_accuracy)