
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
from scipy.stats import randint
import graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, RFE
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


def featureAnalysisSequentialSelector(data, n):
        
    X = data.drop(['user','session', 'task', 'iteration'], axis=1)
    y = data['user']
    originalAccuracy = featureAnalysis(data)[0]
    
    totalFeatures = X.columns

    rfc = RandomForestClassifier()
    selector = SequentialFeatureSelector(rfc, cv=3, scoring='accuracy', direction="forward", n_features_to_select=n)
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
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        main_feature_pred = rf.predict(X_test)
        main_feature_accuracy = accuracy_score(y_test, main_feature_pred)
        accuracyList.append(main_feature_accuracy)
        main_feature_precision = precision_score(y_test, main_feature_pred, average='macro', zero_division=0)
        precisionList.append(main_feature_precision)
        main_feature_recall = recall_score(y_test, main_feature_pred, average='macro', zero_division=0)
        recallList.append(main_feature_recall)
        main_feature_f1_score = f1_score(y_test, main_feature_pred, average='macro', zero_division=0)
        f1ScoreList.append(main_feature_f1_score)
        
    return [n, " ".join(topFeatures) ," ".join(removedFeatures) , originalAccuracy, np.mean(accuracyList), np.mean(precisionList), np.mean(recallList), np.mean(f1ScoreList)]


def featureAnalysisRecursiveCV(data):
    X = data.drop(['user','session', 'task', 'iteration'], axis=1)
    y = data['user']
    originalAccuracy = featureAnalysis(data)[0]

    totalFeatures = X.columns
    print("Total features : %d" %len(totalFeatures))
    # finding an optimal number of features with RFACV
    min_features_to_select = 5
    rfc = RandomForestClassifier()
    selector = RFECV(rfc, step=1, cv=5,min_features_to_select=min_features_to_select)
    selector.fit(X, y)

    X = selector.transform(X)
    removedFeatures = totalFeatures[np.invert(selector.support_)]
    featureImportances = [{'feature':f, 'importance':selector.ranking_[i]} for i,f in enumerate(totalFeatures)]
    removedFeaturesByImportance = [p for p in featureImportances if p["feature"] in removedFeatures]
    removedFeaturesByImportance.sort(key=lambda x: x["importance"], reverse=True)
    print(removedFeaturesByImportance)


    # print(selector.support_)
    # print(selector.ranking_)
    print("Optimal number of features : %d" % selector.n_features_)
    plt.figure()
    plt.xlabel("Nr. of Features")
    plt.ylabel("Accuracy, %")
    
    plt.plot(range(min_features_to_select,len(selector.cv_results_["mean_test_score"]) +min_features_to_select),selector.cv_results_["mean_test_score"])
    plt.show()

    accuracyList = []
    precisionList = []
    recallList = []
    f1ScoreList = []

    for _ in range(10):
        print("iteration")
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.4)
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        main_feature_pred = rf.predict(X_test)
        main_feature_accuracy = accuracy_score(y_test, main_feature_pred)
        accuracyList.append(main_feature_accuracy)
        main_feature_precision = precision_score(y_test, main_feature_pred, average='macro', zero_division=0)
        precisionList.append(main_feature_precision)
        main_feature_recall = recall_score(y_test, main_feature_pred, average='macro', zero_division=0)
        recallList.append(main_feature_recall)
        main_feature_f1_score = f1_score(y_test, main_feature_pred, average='macro', zero_division=0)
        f1ScoreList.append(main_feature_f1_score)

    print(np.mean(accuracyList))
    return [len(selector.get_feature_names_out(input_features=totalFeatures)), " ".join([x["feature"] for x in removedFeaturesByImportance]) , originalAccuracy, np.mean(accuracyList), np.mean(precisionList), np.mean(recallList), np.mean(f1ScoreList)]




def featureAnalysisTask3(data):
    accuracyList = []
    precisionList = []
    recallList = []
    f1ScoreList = []
    for _ in range(10):
        # return [accuracy, precision, recall, f1score] for each feature
        X = data.drop(['user'], axis=1)
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

def featureAnalysisSequentialSelectorTask3(data, n):
        
    X = data.drop(['user'], axis=1)
    y = data['user']
    originalAccuracy = featureAnalysis(data)[0]
    
    totalFeatures = X.columns

    rfc = RandomForestClassifier()
    selector = SequentialFeatureSelector(rfc, cv=3, scoring='accuracy', direction="forward", n_features_to_select=n)
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
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        main_feature_pred = rf.predict(X_test)
        main_feature_accuracy = accuracy_score(y_test, main_feature_pred)
        accuracyList.append(main_feature_accuracy)
        main_feature_precision = precision_score(y_test, main_feature_pred, average='macro', zero_division=0)
        precisionList.append(main_feature_precision)
        main_feature_recall = recall_score(y_test, main_feature_pred, average='macro', zero_division=0)
        recallList.append(main_feature_recall)
        main_feature_f1_score = f1_score(y_test, main_feature_pred, average='macro', zero_division=0)
        f1ScoreList.append(main_feature_f1_score)
        
    return [n, " ".join(topFeatures) ," ".join(removedFeatures) , originalAccuracy, np.mean(accuracyList), np.mean(precisionList), np.mean(recallList), np.mean(f1ScoreList)]


def featureAnalysisRecursiveCVTask3(data):
    X = data.drop(['user'], axis=1)
    y = data['user']
    originalAccuracy = featureAnalysis(data)[0]

    totalFeatures = X.columns
    print("Total features : %d" %len(totalFeatures))
    # finding an optimal number of features with RFACV
    min_features_to_select = 5
    rfc = RandomForestClassifier()
    selector = RFECV(rfc, step=1, cv=5,min_features_to_select=min_features_to_select)
    selector.fit(X, y)

    X = selector.transform(X)
    removedFeatures = totalFeatures[np.invert(selector.support_)]
    featureImportances = [{'feature':f, 'importance':selector.ranking_[i]} for i,f in enumerate(totalFeatures)]
    removedFeaturesByImportance = [p for p in featureImportances if p["feature"] in removedFeatures]
    removedFeaturesByImportance.sort(key=lambda x: x["importance"], reverse=True)
    print(removedFeaturesByImportance)


    # print(selector.support_)
    # print(selector.ranking_)
    print("Optimal number of features : %d" % selector.n_features_)
    plt.figure()
    plt.xlabel("Nr. of Features")
    plt.ylabel("Accuracy, %")
    
    plt.plot(range(min_features_to_select,len(selector.cv_results_["mean_test_score"]) +min_features_to_select),selector.cv_results_["mean_test_score"])
    plt.show()

    accuracyList = []
    precisionList = []
    recallList = []
    f1ScoreList = []

    for _ in range(10):
        print("iteration")
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.4)
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        main_feature_pred = rf.predict(X_test)
        main_feature_accuracy = accuracy_score(y_test, main_feature_pred)
        accuracyList.append(main_feature_accuracy)
        main_feature_precision = precision_score(y_test, main_feature_pred, average='macro', zero_division=0)
        precisionList.append(main_feature_precision)
        main_feature_recall = recall_score(y_test, main_feature_pred, average='macro', zero_division=0)
        recallList.append(main_feature_recall)
        main_feature_f1_score = f1_score(y_test, main_feature_pred, average='macro', zero_division=0)
        f1ScoreList.append(main_feature_f1_score)

    print(np.mean(accuracyList))
    return [len(selector.get_feature_names_out(input_features=totalFeatures)), " ".join([x["feature"] for x in removedFeaturesByImportance]) , originalAccuracy, np.mean(accuracyList), np.mean(precisionList), np.mean(recallList), np.mean(f1ScoreList)]


