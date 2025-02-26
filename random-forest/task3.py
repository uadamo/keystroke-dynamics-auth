"mean_E_value","mean_A_value","mean_T_value","mean_O_value","mean_I_value","mean_N_value","mean_S_value","mean_R_value","mean_L_value","avg_typing_speed","avg_pause_time","free_typing_accuracy","shiftLeft","shiftRight","CapsLock","CommaCount"

# Data Processing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import seaborn as sns
import graphviz
from functools import reduce

import xlwt 
from xlwt import Workbook 
  

task3_data = pd.read_csv('./csv_files/task3_features.csv')



# def xlReportNoSelection():
#     wb = xlwt.Workbook() 

#     # single features - no selection
#     ws_single = wb.add_sheet('task3-features')
#     ws_single.write(0,1,"accuracy")
#     ws_single.write(0,2,"precision")
#     ws_single.write(0,3,"recall")
#     ws_single.write(0,4,"f1-score")

#     eval= [featureAnalysisTask3(task3_data)]
    
#     ws_single.write(1,0,"All features")
#     for i, values in enumerate(eval):
#         ws_single.write(i+1,1,values[0])
    
#     wb.save("task3_rf_no_selection.xls")  

# # xlReportNoSelection()






def featureAnalysisRecursiveCVTask3(data):
    X = data.drop(['user'], axis=1)
    y = data['user']

    totalFeatures = X.columns
    print("Total features : %d" %len(totalFeatures))
    # finding an optimal number of features with RFACV
    min_features_to_select = 5
    rfc = RandomForestClassifier()
    selector = RFECV(rfc, step=1, cv=2,min_features_to_select=min_features_to_select)
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
    return [len(selector.get_feature_names_out(input_features=totalFeatures)), " ".join([x["feature"] for x in removedFeaturesByImportance]) , np.mean(accuracyList), np.mean(precisionList), np.mean(recallList), np.mean(f1ScoreList)]




# def xlReportSequentialSelection():
#     wb = xlwt.Workbook() 

#     # single features - no selection
#     ws = wb.add_sheet('task3-rf-sequential')

#     ws.write(0,1, "nr. of best features")
#     ws.write(0,2, "top features")
#     ws.write(0,3, "removed features")
#     ws.write(0,4,"accuracy (before filtering)")
#     ws.write(0,5,"accuracy")
#     ws.write(0,6,"precision")
#     ws.write(0,7,"recall")
#     ws.write(0,8,"f1-score")

#     # featureAnalysisRecursiveCV(UD_non_temporal_and_statistic_data)

#     feat = [featureAnalysisSequentialSelectorTask3(task3_data, 5)]

#     for i, values in enumerate(feat):
#         ws.write(i+1,1,values[0])
#         ws.write(i+1,2,values[1])
#         ws.write(i+1,3,values[2])
#         ws.write(i+1,4,values[3])
#         ws.write(i+1,5,values[4])
#         ws.write(i+1,6,values[5])
#         ws.write(i+1,7,values[6])
#         ws.write(i+1,8,values[7])
    
#     wb.save("task3_rf_sequential.xls")  

# xlReportSequentialSelection()

def featureAnalysisSequentialSelectorTask3(data, n):
        
    X = data.drop(['user'], axis=1)
    y = data['user']
    
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
        
    return [n, " ".join(topFeatures) ," ".join(removedFeatures) , np.mean(accuracyList), np.mean(precisionList), np.mean(recallList), np.mean(f1ScoreList)]


def xlReportRecursiveCVSelection():
    wb = xlwt.Workbook() 

    # single features - no selection
    ws = wb.add_sheet('task3-rf-recursive-CV')

    ws.write(0,1, "nr. of best features")
    ws.write(0,2, "removed features (least to most important)")
    ws.write(0,3,"accuracy (before filtering)")
    ws.write(0,4,"accuracy")
    ws.write(0,5,"precision")
    ws.write(0,6,"recall")
    ws.write(0,7,"f1-score")


    feat = [featureAnalysisRecursiveCVTask3(task3_data)]

    for i, values in enumerate(feat):
        ws.write(i+1,1,values[0])
        ws.write(i+1,2,values[1])
        ws.write(i+1,3,values[2])
        ws.write(i+1,4,values[3])
        ws.write(i+1,5,values[4])
        ws.write(i+1,6,values[5])
    
    wb.save("task3_rf_recursive-CV.xls")  

xlReportRecursiveCVSelection()