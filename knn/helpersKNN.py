
# Data Processing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, SelectFromModel, SequentialFeatureSelector
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier


def featureAnalysis(data):
    accuracyList = []
    precisionList = []
    recallList = []
    f1ScoreList = []
    X = data.drop(['user','session', 'task', 'iteration'], axis=1)
    y = data['user']
    for _ in range(10):
  
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
        clf = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=selected_k, metric="euclidean")).fit(X_train, y_train)
        main_feature_pred = clf.predict(X_test)
        #best value - 1, worst - 0
        main_feature_accuracy = accuracy_score(y_test, main_feature_pred)
        accuracyList.append(main_feature_accuracy)
        main_feature_precision = precision_score(y_test, main_feature_pred, average='macro', zero_division=0)
        precisionList.append(main_feature_precision)
        main_feature_recall = recall_score(y_test, main_feature_pred, average='macro', zero_division=0)
        recallList.append(main_feature_recall)
        main_feature_f1_score = f1_score(y_test, main_feature_pred, average='macro', zero_division=0)
        f1ScoreList.append(main_feature_f1_score)
    
    return [np.mean(accuracyList), np.mean(precisionList), np.mean(recallList), np.mean(f1ScoreList) ]


def featureAnalysisSequentialSelector(data, n):
        
    X = data.drop(['user','session', 'task', 'iteration'], axis=1)
    y = data['user']
    
    totalFeatures = X.columns

    print(featureAnalysis(data))

    knn = KNeighborsClassifier()
    sfs = SequentialFeatureSelector(knn, cv=5, scoring='accuracy', n_features_to_select=n)
    sfs.fit(X, y)
    X = sfs.transform(X)
    # print(totalFeatures)
    accuracyList = []
    precisionList = []
    recallList = []
    f1ScoreList = []
        
    for _ in range(10):
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
        clf = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=selected_k, metric="euclidean")).fit(X_train, y_train)
        main_feature_pred = clf.predict(X_test)
        #best value - 1, worst - 0
        main_feature_accuracy = accuracy_score(y_test, main_feature_pred)
        accuracyList.append(main_feature_accuracy)
        main_feature_precision = precision_score(y_test, main_feature_pred, average='macro', zero_division=0)
        precisionList.append(main_feature_precision)
        main_feature_recall = recall_score(y_test, main_feature_pred, average='macro', zero_division=0)
        recallList.append(main_feature_recall)
        main_feature_f1_score = f1_score(y_test, main_feature_pred, average='macro', zero_division=0)
        f1ScoreList.append(main_feature_f1_score)
        
    return [n, " ".join(sfs.get_feature_names_out(input_features=totalFeatures)) , np.mean(main_feature_accuracy), np.mean(main_feature_precision), np.mean(main_feature_recall), np.mean(main_feature_f1_score)]

