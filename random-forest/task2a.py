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

merged_DU_reaction_data = pd.merge(UD_data,reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 


merged_UD_accuracy_data = pd.merge(UD_data, accuracy_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_UD_speed_data = pd.merge(UD_data, speed_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_UD_reaction_data = pd.merge(UD_data, reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_UU_accuracy_data = pd.merge(UU_data, accuracy_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_UU_speed_data = pd.merge(UU_data, speed_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_UU_reaction_data = pd.merge(UU_data, reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_DD_accuracy_data = pd.merge(DD_data, accuracy_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_DD_speed_data = pd.merge(DD_data, speed_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_DD_reaction_data = pd.merge(DD_data, reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 
# print(DU_data.target_names)
# print(DU_data.feature_names)

DU_X = DU_data.drop(['user','session', 'task', 'iteration'], axis=1)
DU_y= DU_data['user']

DD_X = DD_data.drop(['user','session', 'task', 'iteration'], axis=1)
DD_y= DD_data['user']

UU_X = UU_data.drop(['user','session', 'task', 'iteration'], axis=1)
UU_y= UU_data['user']

UD_X = UD_data.drop(['user','session', 'task', 'iteration'], axis=1)
UD_y= UD_data['user']

trigraph_X = trigraph_data.drop(['user','session', 'task', 'iteration'], axis=1)
trigraph_y= trigraph_data['user']

merged_DU_accuracy_data_X =  merged_DU_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DU_accuracy_data_y= merged_DU_accuracy_data['user']

merged_DU_speed_data_X =  merged_DU_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DU_speed_data_y= merged_DU_speed_data['user']

merged_DU_reaction_data_X = merged_DU_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DU_reaction_data_y= merged_DU_reaction_data['user']

merged_UD_accuracy_data_X =  merged_UD_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UD_accuracy_data_y= merged_UD_accuracy_data['user']

merged_UD_speed_data_X =  merged_UD_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UD_speed_data_y= merged_UD_speed_data['user']

merged_UD_reaction_data_X =  merged_UD_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UD_reaction_data_y= merged_UD_reaction_data['user']

merged_UU_accuracy_data_X =  merged_UU_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UU_accuracy_data_y= merged_UU_accuracy_data['user']

merged_UU_speed_data_X =  merged_UU_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UU_speed_data_y= merged_UU_speed_data['user']

merged_UU_reaction_data_X =  merged_UU_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UU_reaction_data_y= merged_UU_reaction_data['user']

merged_DD_accuracy_data_X =  merged_DD_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DD_accuracy_data_y= merged_DD_accuracy_data['user']

merged_DD_speed_data_X =  merged_DD_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DD_speed_data_y= merged_DD_speed_data['user']

merged_DD_reaction_data_X =  merged_DD_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DD_reaction_data_y= merged_DD_reaction_data['user']

# Split the data into training and test sets
DU_X_train, DU_X_test, DU_y_train, DU_y_test = train_test_split(DU_X, DU_y, shuffle=True, test_size=0.4)

DD_X_train, DD_X_test, DD_y_train, DD_y_test = train_test_split(DD_X, DD_y, shuffle=True, test_size=0.4)

UU_X_train, UU_X_test, UU_y_train, UU_y_test = train_test_split(UU_X, UU_y,  shuffle=True, test_size=0.4)

UD_X_train, UD_X_test, UD_y_train, UD_y_test = train_test_split(UD_X, UD_y,shuffle=True, test_size=0.4)

trigraph_X_train, trigraph_X_test, trigraph_y_train, trigraph_y_test = train_test_split(trigraph_X, trigraph_y, shuffle=True, test_size=0.4)

merged_DU_accuracy_data_X_train, merged_DU_accuracy_data_X_test, merged_DU_accuracy_data_y_train, merged_DU_accuracy_data_y_test = train_test_split(merged_DU_accuracy_data_X, merged_DU_accuracy_data_y, shuffle=True, test_size=0.4)

merged_DU_speed_data_X_train, merged_DU_speed_data_X_test, merged_DU_speed_data_y_train, merged_DU_speed_data_y_test = train_test_split(merged_DU_speed_data_X, merged_DU_speed_data_y, shuffle=True, test_size=0.4)

merged_DU_reaction_data_X_train, merged_DU_reaction_data_X_test, merged_DU_reaction_data_y_train, merged_DU_reaction_data_y_test = train_test_split(merged_DU_reaction_data_X, merged_DU_reaction_data_y, shuffle=True, test_size=0.4)

merged_UD_accuracy_data_X_train, merged_UD_accuracy_data_X_test, merged_UD_accuracy_data_y_train, merged_UD_accuracy_data_y_test = train_test_split(merged_UD_accuracy_data_X, merged_UD_accuracy_data_y, shuffle=True, test_size=0.4)

merged_UD_speed_data_X_train, merged_UD_speed_data_X_test, merged_UD_speed_data_y_train, merged_UD_speed_data_y_test = train_test_split(merged_UD_speed_data_X, merged_UD_speed_data_y, shuffle=True, test_size=0.4)

merged_UD_reaction_data_X_train, merged_UD_reaction_data_X_test, merged_UD_reaction_data_y_train, merged_UD_reaction_data_y_test = train_test_split(merged_UD_reaction_data_X, merged_UD_reaction_data_y, shuffle=True, test_size=0.4)

merged_UU_accuracy_data_X_train, merged_UU_accuracy_data_X_test, merged_UU_accuracy_data_y_train, merged_UU_accuracy_data_y_test = train_test_split(merged_UU_accuracy_data_X, merged_UU_accuracy_data_y, shuffle=True, test_size=0.4)

merged_UU_speed_data_X_train, merged_UU_speed_data_X_test, merged_UU_speed_data_y_train, merged_UU_speed_data_y_test = train_test_split(merged_UU_speed_data_X, merged_UU_speed_data_y, shuffle=True, test_size=0.4)

merged_UU_reaction_data_X_train, merged_UU_reaction_data_X_test, merged_UU_reaction_data_y_train, merged_UU_reaction_data_y_test = train_test_split(merged_UU_reaction_data_X, merged_UU_reaction_data_y, shuffle=True, test_size=0.4)

merged_DD_accuracy_data_X_train, merged_DD_accuracy_data_X_test, merged_DD_accuracy_data_y_train, merged_DD_accuracy_data_y_test = train_test_split(merged_DD_accuracy_data_X, merged_DD_accuracy_data_y, shuffle=True, test_size=0.4)

merged_DD_speed_data_X_train, merged_DD_speed_data_X_test, merged_DD_speed_data_y_train, merged_DD_speed_data_y_test = train_test_split(merged_DD_speed_data_X, merged_DD_speed_data_y, shuffle=True, test_size=0.4)

merged_DD_reaction_data_X_train, merged_DD_reaction_data_X_test, merged_DD_reaction_data_y_train, merged_DD_reaction_data_y_test = train_test_split(merged_DD_reaction_data_X, merged_DD_reaction_data_y, shuffle=True, test_size=0.4)


rf1 = RandomForestClassifier()
rf1.fit(DU_X_train, DU_y_train)

rf2 = RandomForestClassifier()
rf2.fit(DD_X_train, DD_y_train)

rf3 = RandomForestClassifier()
rf3.fit(UU_X_train, UU_y_train)

rf4 = RandomForestClassifier()
rf4.fit(UD_X_train, UD_y_train)

rf5 = RandomForestClassifier()
rf5.fit(trigraph_X_train, trigraph_y_train)

rf6 = RandomForestClassifier()
rf6.fit(merged_DU_accuracy_data_X_train, merged_DU_accuracy_data_y_train)

rf7 = RandomForestClassifier()
rf7.fit(merged_DU_speed_data_X_train, merged_DU_speed_data_y_train)

rf13 = RandomForestClassifier()
rf13.fit(merged_DU_reaction_data_X_train, merged_DU_reaction_data_y_train)

rf9 = RandomForestClassifier()
rf9.fit(merged_UD_accuracy_data_X_train, merged_UD_accuracy_data_y_train)

rf10 = RandomForestClassifier()
rf10.fit(merged_UD_speed_data_X_train, merged_UD_speed_data_y_train)

rf12 = RandomForestClassifier()
rf12.fit(merged_UD_reaction_data_X_train, merged_UD_reaction_data_y_train)

rf14 = RandomForestClassifier()
rf14.fit(merged_UU_accuracy_data_X_train, merged_UU_accuracy_data_y_train)

rf15 = RandomForestClassifier()
rf15.fit(merged_UU_speed_data_X_train, merged_UU_speed_data_y_train)

rf17 = RandomForestClassifier()
rf17.fit(merged_UU_reaction_data_X_train, merged_UU_reaction_data_y_train)

rf18 = RandomForestClassifier()
rf18.fit(merged_DD_accuracy_data_X_train, merged_DD_accuracy_data_y_train)

rf19 = RandomForestClassifier()
rf19.fit(merged_DD_speed_data_X_train, merged_DD_speed_data_y_train)

rf21 = RandomForestClassifier()
rf21.fit(merged_DD_reaction_data_X_train, merged_DD_reaction_data_y_train)

# False negatives, False posives, ERR, etc
# make a curve

DU_prediction = rf1.predict(DU_X_test)
DU_accuracy = accuracy_score(DU_y_test, DU_prediction)

DD_prediction = rf2.predict(DD_X_test)
DD_accuracy = accuracy_score(DD_y_test, DD_prediction)

UU_prediction = rf3.predict(UU_X_test)
UU_accuracy = accuracy_score(UU_y_test, UU_prediction)

UD_prediction = rf4.predict(UD_X_test)
UD_accuracy = accuracy_score(UD_y_test, UD_prediction)

trigraph_prediction = rf5.predict(trigraph_X_test)
trigraph_accuracy = accuracy_score(trigraph_y_test, trigraph_prediction)

merged_DU_accuracy_data_prediction = rf6.predict(merged_DU_accuracy_data_X_test)
merged_DU_accuracy_data_accuracy = accuracy_score(merged_DU_accuracy_data_y_test, merged_DU_accuracy_data_prediction)

merged_DU_speed_data_prediction = rf7.predict(merged_DU_speed_data_X_test)
merged_DU_speed_data_accuracy = accuracy_score(merged_DU_speed_data_y_test, merged_DU_speed_data_prediction)

merged_DU_reaction_data_prediction = rf13.predict(merged_DU_reaction_data_X_test)
merged_DU_reaction_data_accuracy = accuracy_score(merged_DU_reaction_data_y_test, merged_DU_reaction_data_prediction)

merged_UD_accuracy_data_prediction = rf9.predict(merged_UD_accuracy_data_X_test)
merged_UD_accuracy_data_accuracy = accuracy_score(merged_UD_accuracy_data_y_test, merged_UD_accuracy_data_prediction)

merged_UD_speed_data_prediction = rf10.predict(merged_UD_speed_data_X_test)
merged_UD_speed_data_accuracy = accuracy_score(merged_UD_speed_data_y_test, merged_UD_speed_data_prediction)

merged_UD_reaction_data_prediction = rf12.predict(merged_UD_reaction_data_X_test)
merged_UD_reaction_data_accuracy = accuracy_score(merged_UD_reaction_data_y_test, merged_UD_reaction_data_prediction)

merged_UU_accuracy_data_prediction = rf14.predict(merged_UU_accuracy_data_X_test)
merged_UU_accuracy_data_accuracy = accuracy_score(merged_UU_accuracy_data_y_test, merged_UU_accuracy_data_prediction)

merged_UU_speed_data_prediction = rf15.predict(merged_UU_speed_data_X_test)
merged_UU_speed_data_accuracy = accuracy_score(merged_UU_speed_data_y_test, merged_UU_speed_data_prediction)

merged_UU_reaction_data_prediction = rf17.predict(merged_UU_reaction_data_X_test)
merged_UU_reaction_data_accuracy = accuracy_score(merged_UU_reaction_data_y_test, merged_UU_reaction_data_prediction)

merged_DD_accuracy_data_prediction = rf18.predict(merged_DD_accuracy_data_X_test)
merged_DD_accuracy_data_accuracy = accuracy_score(merged_DD_accuracy_data_y_test, merged_DD_accuracy_data_prediction)

merged_DD_speed_data_prediction = rf19.predict(merged_DD_speed_data_X_test)
merged_DD_speed_data_accuracy = accuracy_score(merged_DD_speed_data_y_test, merged_DD_speed_data_prediction)

merged_DD_reaction_data_prediction = rf21.predict(merged_DD_reaction_data_X_test)
merged_DD_reaction_data_accuracy = accuracy_score(merged_DD_reaction_data_y_test, merged_DD_reaction_data_prediction)

# plt.figure(figsize=(16, 8))

# merged_UD_accuracy_data_corr = merged_UD_accuracy_data.corr()
# sns.heatmap(merged_UD_accuracy_data_corr, annot=True, square=True, fmt='0.2f')

# plt.show()

def accuracyReport(mainFeatureName, mainFeatureAccuracy, mergedAccuracy, mergedSpeed, mergedReaction):
    mergedMap = {"typing accuracy": mergedAccuracy, "typing speed": mergedSpeed,"reaction": mergedReaction }

    print(mainFeatureName + " accuracy :", mainFeatureAccuracy)
    for key, value in mergedMap.items():
        if(value - mainFeatureAccuracy >= 0.08):
            print(f"{mainFeatureName} merged with {key} accuracy: {value} STRONGLY IMPROVED original")
        elif(value - mainFeatureAccuracy < 0.08 and value - mainFeatureAccuracy > 0.04 and value > mainFeatureAccuracy):
            print(f"{mainFeatureName} merged with {key} accuracy: {value} improved original")
        else:
            print(f"{mainFeatureName} merged with {key} accuracy: {value}")
        
    print("\n")


accuracyReport("DU", DU_accuracy, merged_DU_accuracy_data_accuracy, merged_DU_speed_data_accuracy, merged_DU_reaction_data_accuracy)

accuracyReport("UD", UD_accuracy, merged_UD_accuracy_data_accuracy, merged_UD_speed_data_accuracy, merged_UD_reaction_data_accuracy)

accuracyReport("UU", UU_accuracy, merged_UU_accuracy_data_accuracy, merged_UU_speed_data_accuracy, merged_UU_reaction_data_accuracy)

accuracyReport("DD", DD_accuracy, merged_DD_accuracy_data_accuracy, merged_DD_speed_data_accuracy, merged_DD_reaction_data_accuracy)
