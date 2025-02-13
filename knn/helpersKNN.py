# Data Processing
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import seaborn as sns
import graphviz
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

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


# run function 10 times, get average metrics
# accuracy, 


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
        scaler_main = StandardScaler()
        scaler_main.fit(X_train)
        X_train = scaler_main.transform(X_train)
        X_test = scaler_main.transform(X_test)
        scores_main = {}
        scores_main_list = []
        for k in range(1,15):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores_main[k] = metrics.accuracy_score(y_test,y_pred)
            scores_main_list.append(metrics.accuracy_score(y_test,y_pred))

        selected_index= np.argmax(scores_main_list)
        selected_k = range(1,30)[selected_index]
        knn = KNeighborsClassifier(n_neighbors=selected_k)
        knn.fit(X_train, y_train)
        main_feature_pred = knn.predict(X_test)
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


