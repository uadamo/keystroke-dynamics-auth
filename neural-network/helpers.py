
# Data Processing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from IPython.display import Image
import seaborn as sns
import graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from ast import literal_eval


def featureAnalysis(data):
    accuracyList = []
    precisionList = []
    recallList = []
    f1ScoreList = []

    scaler = StandardScaler()
    X = data.drop(['user','session', 'task', 'iteration'], axis=1)
    y = data['user']
    X = scaler.fit(X).transform(X)

    for _ in range(10):
        # return [accuracy, precision, recall, f1score] for each feature

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.4)
        clf = OneVsOneClassifier(MLPClassifier( solver="lbfgs", max_iter=5000)).fit(X_train, y_train)

        main_feature_pred = clf.predict(X_test)
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

    print(n)
        
    X = data.drop(['user','session', 'task', 'iteration'], axis=1)
    y = data['user']
    originalAccuracy = featureAnalysis(data)[0]
    
    totalFeatures = X.columns

    # ovo = OneVsOneClassifier(svm).fit(X_train, y_train)
    # main_feature_pred = ovo.predict(X_test)
    X = data.drop(['user','session', 'task', 'iteration'], axis=1)
    y = data['user']
    scaler = StandardScaler()
    X = scaler.fit(X).transform(X)
    mlp =  MLPClassifier( solver="lbfgs", random_state=42, max_iter=5000000000)

    selector = SequentialFeatureSelector(mlp, cv=2, scoring='accuracy', direction="forward", n_features_to_select=n)
    selector.fit(X, y)

    topFeatures = totalFeatures[selector.support_]
    print(topFeatures)
    removedFeatures = totalFeatures[np.invert(selector.support_)]
    print(removedFeatures)

    X = selector.transform(X)
    # print(totalFeatures)
    accuracyList = []
    precisionList = []
    recallList = []
    f1ScoreList = []
        
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.4)
        mlp=  make_pipeline(StandardScaler(), MLPClassifier( solver="lbfgs", random_state=42, max_iter=5000000000))
        mlp.fit(X_train, y_train)
        main_feature_pred = mlp.predict(X_test)
        main_feature_accuracy = accuracy_score(y_test, main_feature_pred)
        accuracyList.append(main_feature_accuracy)
        main_feature_precision = precision_score(y_test, main_feature_pred, average='macro', zero_division=0)
        precisionList.append(main_feature_precision)
        main_feature_recall = recall_score(y_test, main_feature_pred, average='macro', zero_division=0)
        recallList.append(main_feature_recall)
        main_feature_f1_score = f1_score(y_test, main_feature_pred, average='macro', zero_division=0)
        f1ScoreList.append(main_feature_f1_score)
        
    return [n, " ".join(removedFeatures) , originalAccuracy, np.mean(accuracyList), np.mean(precisionList), np.mean(recallList), np.mean(f1ScoreList)]

