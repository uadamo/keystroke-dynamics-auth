
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
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier


def featureAnalysis(data):
    accuracyList = []
    precisionList = []
    recallList = []
    f1ScoreList = []
    for _ in range(10):
        # return [accuracy, precision, recall, f1score] for each feature
        X = data.drop(['user','session', 'task', 'iteration'], axis=1)
        y = data['user']
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.4)

        rf = RandomForestClassifier()

        # ovo = OneVsOneClassifier(rf).fit(X_train, y_train)
        # main_feature_pred = ovo.predict(X_test)
        rf.fit(X_train, y_train)
        main_feature_pred = rf.predict(X_test)


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


def featureAnalysisSequentialSelector(data):
        
    X = data.drop(['user','session', 'task', 'iteration'], axis=1)
    y = data['user']

    rfc = RandomForestClassifier()
    sfs = SequentialFeatureSelector(rfc, scoring='accuracy',)
    sfs.fit(X, y)
    print(sfs.get_feature_names_out())
    
    accuracyList = []
    precisionList = []
    recallList = []
    f1ScoreList = []

    for _ in range(10):

        X_train, X_test, y_train, y_test = train_test_split(sfs.transform(X), y, shuffle=True, test_size=0.4)
        rf = RandomForestClassifier()
        # ovo = OneVsOneClassifier(rf).fit(X_train, y_train)
        # main_feature_pred = ovo.predict(X_test)
        main_feature_pred = rf.fit(X_train, y_train)

        #best value - 1, worst - 0
        main_feature_accuracy = accuracy_score(y_test, main_feature_pred)
        accuracyList.append(main_feature_accuracy)
        main_feature_precision = precision_score(y_test, main_feature_pred, average='macro', zero_division=0)
        precisionList.append(main_feature_precision)
        main_feature_recall = recall_score(y_test, main_feature_pred, average='macro', zero_division=0)
        recallList.append(main_feature_recall)
        main_feature_f1_score = f1_score(y_test, main_feature_pred, average='macro', zero_division=0)
        f1ScoreList.append(main_feature_f1_score)

        cnf_matrix = confusion_matrix(y_true = y_test, y_pred=main_feature_pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        
        FPR = FP/(FP+TN)
        FNR = FN/(TP+FN)
        
        ACC = (TP+TN)/(TP+FP+FN+TN)

    
    return [np.mean(main_feature_accuracy), np.mean(main_feature_precision), np.mean(main_feature_recall), np.mean(main_feature_f1_score) ]


