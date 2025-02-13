
# Data Processing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import graphviz
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def accuracyReport(mainFeatureName, mainFeatureAccuracy, mergedAccuracy, mergedSpeed, mergedKeyPreference, mergedReaction):
    mergedMap = {"typing accuracy": mergedAccuracy, "typing speed": mergedSpeed, "key preference": mergedKeyPreference,"reaction": mergedReaction }

    print(mainFeatureName + ",", mainFeatureAccuracy)
    for key, value in mergedMap.items():
        print(f"{mainFeatureName} + {key} , {value}")

def accuracyReportNoKp(mainFeatureName, mainFeatureAccuracy, mergedAccuracy, mergedSpeed, mergedReaction):
    mergedMap = {"typing accuracy": mergedAccuracy, "typing speed": mergedSpeed,"reaction": mergedReaction }

    print(mainFeatureName + ",", mainFeatureAccuracy)
    for key, value in mergedMap.items():
        print(f"{mainFeatureName}+{key}, {value}")


def featureAnalysisSingleData(data):
    accuracyList = []
    precisionList = []
    recallList = []
    f1ScoreList = []
    for _ in range(10):
        # return [accuracy, precision, recall, f1score] for each feature
        X = data.drop(['user','session', 'task', 'iteration'], axis=1)
        y = data['user']
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.4)

        rf1 = RandomForestClassifier()
        rf1.fit(X_train, y_train)

        main_feature_pred = rf1.predict(X_test)

        #best value - 1, worst - 0
        main_feature_accuracy = accuracy_score(y_test, main_feature_pred)
        accuracyList.append(main_feature_accuracy)
        main_feature_precision = precision_score(y_test, main_feature_pred, average='macro', zero_division=0)
        precisionList.append(main_feature_precision)
        main_feature_recall = recall_score(y_test, main_feature_pred, average='macro', zero_division=0)
        recallList.append(main_feature_recall)
        main_feature_f1_score = f1_score(y_test, main_feature_pred, average='macro', zero_division=0)
        f1ScoreList.append(main_feature_f1_score)
    
    print(accuracyList)
    return [np.mean(main_feature_accuracy), np.mean(main_feature_precision), np.mean(main_feature_recall), np.mean(main_feature_f1_score) ]



def featureAccuracyAnalysis(DU_data, DD_data, UU_data, UD_data, trigraph_data, accuracy_data, speed_data, keyPreference_data, reaction_data):
    data_options = dict(DU = DU_data, DD= DD_data, UU= UU_data, UD= UD_data, trigraph= trigraph_data )

    for key, value in data_options.items():
        # defining merged arrays
        merged_accuracy_data = pd.merge(value , accuracy_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 
        merged_speed_data = pd.merge(value, speed_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 
        merged_keyPreference_data = pd.merge(value, keyPreference_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 
        merged_reaction_data = pd.merge(value,reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner')
        
        X = value.drop(['user','session', 'task', 'iteration'], axis=1)
        y = value['user']

        merged_accuracy_data_X =  merged_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
        merged_accuracy_data_y= merged_accuracy_data['user']
        merged_speed_data_X =  merged_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
        merged_speed_data_y= merged_speed_data['user']
        merged_keyPreference_data_X = merged_keyPreference_data.drop(['user','session', 'task', 'iteration'], axis=1)
        merged_keyPreference_data_y= merged_keyPreference_data['user']
        merged_reaction_data_X = merged_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
        merged_reaction_data_y= merged_reaction_data['user']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.4)
        merged_accuracy_data_X_train, merged_accuracy_data_X_test, merged_accuracy_data_y_train, merged_accuracy_data_y_test = train_test_split(merged_accuracy_data_X, merged_accuracy_data_y, shuffle=True, test_size=0.4)
        merged_speed_data_X_train, merged_speed_data_X_test, merged_speed_data_y_train, merged_speed_data_y_test = train_test_split(merged_speed_data_X, merged_speed_data_y, shuffle=True, test_size=0.4)
        merged_keyPreference_data_X_train, merged_keyPreference_data_X_test, merged_keyPreference_data_y_train, merged_keyPreference_data_y_test = train_test_split(merged_keyPreference_data_X, merged_keyPreference_data_y, shuffle=True, test_size=0.4)
        merged_reaction_data_X_train, merged_reaction_data_X_test, merged_reaction_data_y_train, merged_reaction_data_y_test = train_test_split(merged_reaction_data_X, merged_reaction_data_y, shuffle=True, test_size=0.4)

        rf1 = RandomForestClassifier()
        rf1.fit(X_train, y_train)

        rf2 = RandomForestClassifier()
        rf2.fit(merged_accuracy_data_X_train, merged_accuracy_data_y_train)

        rf3 = RandomForestClassifier()
        rf3.fit(merged_speed_data_X_train, merged_speed_data_y_train)

        rf4 = RandomForestClassifier()
        rf4.fit(merged_keyPreference_data_X_train, merged_keyPreference_data_y_train)

        rf5 = RandomForestClassifier()
        rf5.fit(merged_reaction_data_X_train, merged_reaction_data_y_train)

        main_feature_prediction = rf1.predict(X_test)
        main_feature_accuracy = accuracy_score(y_test, main_feature_prediction)

        merged_accuracy_data_prediction = rf2.predict(merged_accuracy_data_X_test)
        merged_accuracy_data_accuracy = accuracy_score(merged_accuracy_data_y_test, merged_accuracy_data_prediction)
        
        merged_speed_data_prediction = rf3.predict(merged_speed_data_X_test)
        merged_speed_data_accuracy = accuracy_score(merged_speed_data_y_test, merged_speed_data_prediction)
        
        merged_keyPreference_data_prediction = rf4.predict(merged_keyPreference_data_X_test)
        merged_keyPreference_data_accuracy = accuracy_score(merged_keyPreference_data_y_test, merged_keyPreference_data_prediction)
        
        merged_reaction_data_prediction = rf5.predict(merged_reaction_data_X_test)
        merged_reaction_data_accuracy = accuracy_score(merged_reaction_data_y_test, merged_reaction_data_prediction)

        accuracyReport(key, main_feature_accuracy, merged_accuracy_data_accuracy, merged_speed_data_accuracy, merged_keyPreference_data_accuracy, merged_reaction_data_accuracy)

def featureAccuracyAnalysisNoKp(DU_data, DD_data, UU_data, UD_data, trigraph_data, accuracy_data, speed_data, reaction_data):
    data_options = dict(DU = DU_data, DD= DD_data, UU= UU_data, UD= UD_data, trigraph= trigraph_data )

    for key, value in data_options.items():
        # defining merged arrays
        merged_accuracy_data = pd.merge(value , accuracy_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 
        merged_speed_data = pd.merge(value, speed_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 
        merged_reaction_data = pd.merge(value,reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner')
        
        X = value.drop(['user','session', 'task', 'iteration'], axis=1)
        y = value['user']

        merged_accuracy_data_X =  merged_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
        merged_accuracy_data_y= merged_accuracy_data['user']
        merged_speed_data_X =  merged_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
        merged_speed_data_y= merged_speed_data['user']
        merged_reaction_data_X = merged_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
        merged_reaction_data_y= merged_reaction_data['user']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.4)
        merged_accuracy_data_X_train, merged_accuracy_data_X_test, merged_accuracy_data_y_train, merged_accuracy_data_y_test = train_test_split(merged_accuracy_data_X, merged_accuracy_data_y, shuffle=True, test_size=0.4)
        merged_speed_data_X_train, merged_speed_data_X_test, merged_speed_data_y_train, merged_speed_data_y_test = train_test_split(merged_speed_data_X, merged_speed_data_y, shuffle=True, test_size=0.4)
        merged_reaction_data_X_train, merged_reaction_data_X_test, merged_reaction_data_y_train, merged_reaction_data_y_test = train_test_split(merged_reaction_data_X, merged_reaction_data_y, shuffle=True, test_size=0.4)

        rf1 = RandomForestClassifier()
        rf1.fit(X_train, y_train)

        rf2 = RandomForestClassifier()
        rf2.fit(merged_accuracy_data_X_train, merged_accuracy_data_y_train)

        rf3 = RandomForestClassifier()
        rf3.fit(merged_speed_data_X_train, merged_speed_data_y_train)


        rf5 = RandomForestClassifier()
        rf5.fit(merged_reaction_data_X_train, merged_reaction_data_y_train)

        main_feature_prediction = rf1.predict(X_test)
        main_feature_accuracy = accuracy_score(y_test, main_feature_prediction)

        merged_accuracy_data_prediction = rf2.predict(merged_accuracy_data_X_test)
        merged_accuracy_data_accuracy = accuracy_score(merged_accuracy_data_y_test, merged_accuracy_data_prediction)
        
        merged_speed_data_prediction = rf3.predict(merged_speed_data_X_test)
        merged_speed_data_accuracy = accuracy_score(merged_speed_data_y_test, merged_speed_data_prediction)
        
        
        merged_reaction_data_prediction = rf5.predict(merged_reaction_data_X_test)
        merged_reaction_data_accuracy = accuracy_score(merged_reaction_data_y_test, merged_reaction_data_prediction)

        accuracyReportNoKp(key, main_feature_accuracy, merged_accuracy_data_accuracy, merged_speed_data_accuracy, merged_reaction_data_accuracy)