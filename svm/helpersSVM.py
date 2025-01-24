
# Data Processing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from IPython.display import Image
import seaborn as sns
import graphviz
from ast import literal_eval

def accuracyReport(mainFeatureName, mainFeatureAccuracy, mergedAccuracy, mergedSpeed, mergedKeyPreference, mergedReaction):
    mergedMap = {"typing accuracy": mergedAccuracy, "typing speed": mergedSpeed, "key preference": mergedKeyPreference,"reaction": mergedReaction }

    # print(mainFeatureName + ",", mainFeatureAccuracy)
    print(mainFeatureAccuracy)
    for key, value in mergedMap.items():
        # print(f"{mainFeatureName} + {key} , {value}")
        print(value)

def accuracyReportNoKp(mainFeatureName, mainFeatureAccuracy, mergedAccuracy, mergedSpeed, mergedReaction):
    mergedMap = {"typing accuracy": mergedAccuracy, "typing speed": mergedSpeed,"reaction": mergedReaction }

    # print(mainFeatureName + ",", mainFeatureAccuracy)
    print(mainFeatureAccuracy)
    for key, value in mergedMap.items():
        # print(f"{mainFeatureName} + {key} , {value}")
        print(value)

# def confusionMatrixAndScores():
#     FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
#     FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
#     TP = np.diag(confusion_matrix)
#     TN = confusion_matrix.values.sum() - (FP + FN + TP)
#     # Sensitivity, hit rate, recall, or true positive rate
#     TPR = TP/(TP+FN)
#     # Specificity or true negative rate
#     TNR = TN/(TN+FP) 
#     # Precision or positive predictive value
#     PPV = TP/(TP+FP)
#     # # Negative predictive value
#     NPV = TN/(TN+FN)
#     # # Fall out or false positive rate
#     FPR = FP/(FP+TN)
#     # False negative rate
#     FNR = FN/(TP+FN)
#     # False discovery rate
#     FDR = FP/(TP+FP)


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

        clf1 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf1.fit(X_train, y_train)

        clf2 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf2.fit(merged_accuracy_data_X_train, merged_accuracy_data_y_train)

        clf3 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf3.fit(merged_speed_data_X_train, merged_speed_data_y_train)

        clf4 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf4.fit(merged_keyPreference_data_X_train, merged_keyPreference_data_y_train)

        clf5 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf5.fit(merged_reaction_data_X_train, merged_reaction_data_y_train)

        main_feature_prediction = clf1.predict(X_test)
        main_feature_accuracy = accuracy_score(y_test, main_feature_prediction)

        merged_accuracy_data_prediction = clf2.predict(merged_accuracy_data_X_test)
        merged_accuracy_data_accuracy = accuracy_score(merged_accuracy_data_y_test, merged_accuracy_data_prediction)
        
        merged_speed_data_prediction = clf3.predict(merged_speed_data_X_test)
        merged_speed_data_accuracy = accuracy_score(merged_speed_data_y_test, merged_speed_data_prediction)
        
        merged_keyPreference_data_prediction = clf4.predict(merged_keyPreference_data_X_test)
        merged_keyPreference_data_accuracy = accuracy_score(merged_keyPreference_data_y_test, merged_keyPreference_data_prediction)
        
        merged_reaction_data_prediction = clf5.predict(merged_reaction_data_X_test)
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

        clf1 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf1.fit(X_train, y_train)

        clf2 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf2.fit(merged_accuracy_data_X_train, merged_accuracy_data_y_train)

        clf3 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf3.fit(merged_speed_data_X_train, merged_speed_data_y_train)


        clf5 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf5.fit(merged_reaction_data_X_train, merged_reaction_data_y_train)

        main_feature_prediction = clf1.predict(X_test)
        main_feature_accuracy = accuracy_score(y_test, main_feature_prediction)

        merged_accuracy_data_prediction = clf2.predict(merged_accuracy_data_X_test)
        merged_accuracy_data_accuracy = accuracy_score(merged_accuracy_data_y_test, merged_accuracy_data_prediction)
        
        merged_speed_data_prediction = clf3.predict(merged_speed_data_X_test)
        merged_speed_data_accuracy = accuracy_score(merged_speed_data_y_test, merged_speed_data_prediction)
        
        
        merged_reaction_data_prediction = clf5.predict(merged_reaction_data_X_test)
        merged_reaction_data_accuracy = accuracy_score(merged_reaction_data_y_test, merged_reaction_data_prediction)

        accuracyReportNoKp(key, main_feature_accuracy, merged_accuracy_data_accuracy, merged_speed_data_accuracy, merged_reaction_data_accuracy)