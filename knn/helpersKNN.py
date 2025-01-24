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

def confusionMatrixAndClassificationReport(title,test,pred):
    print(title)
    confusion_mtr = metrics.confusion_matrix(test,pred)
    report = metrics.classification_report(test, pred, zero_division=1)


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

        conf_matr = metrics.confusion_matrix(y_test, y_pred)
        result_1 = metrics.classification_report(y_test, y_pred, zero_division=1)
        selected_index= np.argmax(scores_main_list)
        selected_k = range(1,30)[selected_index]
        knn = KNeighborsClassifier(n_neighbors=selected_k)
        knn.fit(X_train, y_train)
        main_feature_pred = knn.predict(X_test)
        main_feature_accuracy = accuracy_score(y_test, main_feature_pred)

        # merged accuracy
        scaler_merged_accuracy_data = StandardScaler()
        scaler_merged_accuracy_data.fit(merged_accuracy_data_X_train)
        merged_accuracy_data_X_train =scaler_merged_accuracy_data.transform(merged_accuracy_data_X_train)
        merged_accuracy_data_X_test = scaler_merged_accuracy_data.transform(merged_accuracy_data_X_test)
        scores_merged_accuracy_data = {}
        scores_merged_accuracy_data_list = []
        for k in range(1,15):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(merged_accuracy_data_X_train, merged_accuracy_data_y_train)
            merged_accuracy_data_y_pred = model.predict(merged_accuracy_data_X_test)
            scores_merged_accuracy_data[k] = metrics.accuracy_score(merged_accuracy_data_y_test,merged_accuracy_data_y_pred)
            scores_merged_accuracy_data_list.append(metrics.accuracy_score(merged_accuracy_data_y_test,merged_accuracy_data_y_pred))
        
        conf_matr_merged_accuracy = metrics.confusion_matrix(merged_accuracy_data_y_test, merged_accuracy_data_y_pred)
        result_1_merged_reaction = metrics.classification_report(merged_accuracy_data_y_test, merged_accuracy_data_y_pred, zero_division=1)
        selected_index_merged_accuracy= np.argmax(scores_merged_accuracy_data_list)
        selected_k_merged_accuracy = range(1,30)[selected_index_merged_accuracy]
        knn_merged_accuracy = KNeighborsClassifier(n_neighbors=selected_k_merged_accuracy)
        knn_merged_accuracy.fit(merged_accuracy_data_X_train, merged_accuracy_data_y_train)
        merged_accuracy_pred = knn_merged_accuracy.predict(merged_accuracy_data_X_test)
        merged_accuracy_data_accuracy = accuracy_score(merged_accuracy_data_y_test, merged_accuracy_pred)

        # merged speed
        scaler_merged_speed_data = StandardScaler()
        scaler_merged_speed_data.fit(merged_speed_data_X_train)
        merged_speed_data_X_train = scaler_merged_speed_data.transform(merged_speed_data_X_train)
        merged_speed_data_X_test = scaler_merged_speed_data.transform(merged_speed_data_X_test)
        scores_merged_speed_data = {}
        scores_merged_speed_data_list = []
        for k in range(1,15):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(merged_speed_data_X_train, merged_speed_data_y_train)
            merged_speed_data_y_pred = model.predict(merged_speed_data_X_test)
            scores_merged_speed_data[k] = metrics.accuracy_score(merged_speed_data_y_test,merged_speed_data_y_pred)
            scores_merged_speed_data_list.append(metrics.accuracy_score(merged_speed_data_y_test,merged_speed_data_y_pred))
        
        conf_matr_merged_speed = metrics.confusion_matrix(merged_speed_data_y_test, merged_speed_data_y_pred)
        result_1_merged_speed = metrics.classification_report(merged_speed_data_y_test, merged_speed_data_y_pred, zero_division=1)
        selected_index_merged_speed= np.argmax(scores_merged_speed_data_list)
        selected_k_merged_speed = range(1,30)[selected_index_merged_speed]
        knn_merged_speed = KNeighborsClassifier(n_neighbors=selected_k_merged_speed)
        knn_merged_speed.fit(merged_speed_data_X_train, merged_speed_data_y_train)
        merged_speed_pred = knn_merged_speed.predict(merged_speed_data_X_test)
        merged_speed_data_accuracy = accuracy_score(merged_speed_data_y_test, merged_speed_pred)

        # merged keypref
        scaler_merged_keyPreference_data = StandardScaler()
        scaler_merged_keyPreference_data.fit(merged_keyPreference_data_X_train)
        merged_keyPreference_data_X_train = scaler_merged_keyPreference_data.transform(merged_keyPreference_data_X_train)
        merged_keyPreference_data_X_test = scaler_merged_keyPreference_data.transform(merged_keyPreference_data_X_test)
        scores_merged_keyPreference_data = {}
        scores_merged_keyPreference_data_list = []
        for k in range(1,15):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(merged_keyPreference_data_X_train, merged_keyPreference_data_y_train)
            merged_keyPreference_data_y_pred = model.predict(merged_keyPreference_data_X_test)
            scores_merged_keyPreference_data[k] = metrics.accuracy_score(merged_keyPreference_data_y_test,merged_keyPreference_data_y_pred)
            scores_merged_keyPreference_data_list.append(metrics.accuracy_score(merged_keyPreference_data_y_test,merged_keyPreference_data_y_pred))
        
        conf_matr_merged_keyPreference = metrics.confusion_matrix(merged_keyPreference_data_y_test, merged_keyPreference_data_y_pred)
        result_1_merged_keyPreference = metrics.classification_report(merged_keyPreference_data_y_test, merged_keyPreference_data_y_pred, zero_division=1)
        selected_index_merged_keyPreference= np.argmax(scores_merged_keyPreference_data_list)
        selected_k_merged_keyPreference = range(1,30)[selected_index_merged_keyPreference]
        knn_merged_keyPreference = KNeighborsClassifier(n_neighbors=selected_k_merged_keyPreference)
        knn_merged_keyPreference.fit(merged_keyPreference_data_X_train, merged_keyPreference_data_y_train)
        merged_keyPreference_pred = knn_merged_keyPreference.predict(merged_keyPreference_data_X_test)
        merged_keyPreference_data_accuracy = accuracy_score(merged_keyPreference_data_y_test, merged_keyPreference_pred)
        # merged reaction
        scaler_merged_reaction_data = StandardScaler()
        scaler_merged_reaction_data.fit(merged_reaction_data_X_train)
        merged_reaction_data_X_train = scaler_merged_reaction_data.transform(merged_reaction_data_X_train)
        merged_reaction_data_X_test = scaler_merged_reaction_data.transform(merged_reaction_data_X_test)
        scores_merged_reaction_data = {}
        scores_merged_reaction_data_list = []
        for k in range(1,15):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(merged_reaction_data_X_train, merged_reaction_data_y_train)
            merged_reaction_data_y_pred = model.predict(merged_reaction_data_X_test)
            scores_merged_reaction_data[k] = metrics.accuracy_score(merged_reaction_data_y_test,merged_reaction_data_y_pred)
            scores_merged_reaction_data_list.append(metrics.accuracy_score(merged_reaction_data_y_test,merged_reaction_data_y_pred))
        
        conf_matr_merged_reaction = metrics.confusion_matrix(merged_reaction_data_y_test, merged_reaction_data_y_pred)
        result_1_merged_reaction = metrics.classification_report(merged_reaction_data_y_test, merged_reaction_data_y_pred, zero_division=1)
        selected_index_merged_reaction= np.argmax(scores_merged_reaction_data_list)
        selected_k_merged_reaction = range(1,30)[selected_index_merged_reaction]
        knn_merged_reaction = KNeighborsClassifier(n_neighbors=selected_k_merged_reaction)
        knn_merged_reaction.fit(merged_reaction_data_X_train, merged_reaction_data_y_train)
        merged_reaction_pred = knn_merged_reaction.predict(merged_reaction_data_X_test)
        merged_reaction_data_accuracy = accuracy_score(merged_reaction_data_y_test, merged_reaction_pred)

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

        conf_matr = metrics.confusion_matrix(y_test, y_pred)
        result_1 = metrics.classification_report(y_test, y_pred, zero_division=1)
        selected_index= np.argmax(scores_main_list)
        selected_k = range(1,30)[selected_index]
        knn = KNeighborsClassifier(n_neighbors=selected_k)
        knn.fit(X_train, y_train)
        main_feature_pred = knn.predict(X_test)
        main_feature_accuracy = accuracy_score(y_test, main_feature_pred)

        # merged accuracy
        scaler_merged_accuracy_data = StandardScaler()
        scaler_merged_accuracy_data.fit(merged_accuracy_data_X_train)
        merged_accuracy_data_X_train =scaler_merged_accuracy_data.transform(merged_accuracy_data_X_train)
        merged_accuracy_data_X_test = scaler_merged_accuracy_data.transform(merged_accuracy_data_X_test)
        scores_merged_accuracy_data = {}
        scores_merged_accuracy_data_list = []
        for k in range(1,15):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(merged_accuracy_data_X_train, merged_accuracy_data_y_train)
            merged_accuracy_data_y_pred = model.predict(merged_accuracy_data_X_test)
            scores_merged_accuracy_data[k] = metrics.accuracy_score(merged_accuracy_data_y_test,merged_accuracy_data_y_pred)
            scores_merged_accuracy_data_list.append(metrics.accuracy_score(merged_accuracy_data_y_test,merged_accuracy_data_y_pred))
        
        conf_matr_merged_accuracy = metrics.confusion_matrix(merged_accuracy_data_y_test, merged_accuracy_data_y_pred)
        result_1_merged_reaction = metrics.classification_report(merged_accuracy_data_y_test, merged_accuracy_data_y_pred, zero_division=1)
        selected_index_merged_accuracy= np.argmax(scores_merged_accuracy_data_list)
        selected_k_merged_accuracy = range(1,30)[selected_index_merged_accuracy]
        knn_merged_accuracy = KNeighborsClassifier(n_neighbors=selected_k_merged_accuracy)
        knn_merged_accuracy.fit(merged_accuracy_data_X_train, merged_accuracy_data_y_train)
        merged_accuracy_pred = knn_merged_accuracy.predict(merged_accuracy_data_X_test)
        merged_accuracy_data_accuracy = accuracy_score(merged_accuracy_data_y_test, merged_accuracy_pred)

        # merged speed
        scaler_merged_speed_data = StandardScaler()
        scaler_merged_speed_data.fit(merged_speed_data_X_train)
        merged_speed_data_X_train = scaler_merged_speed_data.transform(merged_speed_data_X_train)
        merged_speed_data_X_test = scaler_merged_speed_data.transform(merged_speed_data_X_test)
        scores_merged_speed_data = {}
        scores_merged_speed_data_list = []
        for k in range(1,15):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(merged_speed_data_X_train, merged_speed_data_y_train)
            merged_speed_data_y_pred = model.predict(merged_speed_data_X_test)
            scores_merged_speed_data[k] = metrics.accuracy_score(merged_speed_data_y_test,merged_speed_data_y_pred)
            scores_merged_speed_data_list.append(metrics.accuracy_score(merged_speed_data_y_test,merged_speed_data_y_pred))
        
        conf_matr_merged_speed = metrics.confusion_matrix(merged_speed_data_y_test, merged_speed_data_y_pred)
        result_1_merged_speed = metrics.classification_report(merged_speed_data_y_test, merged_speed_data_y_pred, zero_division=1)
        selected_index_merged_speed= np.argmax(scores_merged_speed_data_list)
        selected_k_merged_speed = range(1,30)[selected_index_merged_speed]
        knn_merged_speed = KNeighborsClassifier(n_neighbors=selected_k_merged_speed)
        knn_merged_speed.fit(merged_speed_data_X_train, merged_speed_data_y_train)
        merged_speed_pred = knn_merged_speed.predict(merged_speed_data_X_test)
        merged_speed_data_accuracy = accuracy_score(merged_speed_data_y_test, merged_speed_pred)

        # merged reaction
        scaler_merged_reaction_data = StandardScaler()
        scaler_merged_reaction_data.fit(merged_reaction_data_X_train)
        merged_reaction_data_X_train = scaler_merged_reaction_data.transform(merged_reaction_data_X_train)
        merged_reaction_data_X_test = scaler_merged_reaction_data.transform(merged_reaction_data_X_test)
        scores_merged_reaction_data = {}
        scores_merged_reaction_data_list = []
        for k in range(1,15):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(merged_reaction_data_X_train, merged_reaction_data_y_train)
            merged_reaction_data_y_pred = model.predict(merged_reaction_data_X_test)
            scores_merged_reaction_data[k] = metrics.accuracy_score(merged_reaction_data_y_test,merged_reaction_data_y_pred)
            scores_merged_reaction_data_list.append(metrics.accuracy_score(merged_reaction_data_y_test,merged_reaction_data_y_pred))
        
        conf_matr_merged_reaction = metrics.confusion_matrix(merged_reaction_data_y_test, merged_reaction_data_y_pred)
        result_1_merged_reaction = metrics.classification_report(merged_reaction_data_y_test, merged_reaction_data_y_pred, zero_division=1)
        selected_index_merged_reaction= np.argmax(scores_merged_reaction_data_list)
        selected_k_merged_reaction = range(1,30)[selected_index_merged_reaction]
        knn_merged_reaction = KNeighborsClassifier(n_neighbors=selected_k_merged_reaction)
        knn_merged_reaction.fit(merged_reaction_data_X_train, merged_reaction_data_y_train)
        merged_reaction_pred = knn_merged_reaction.predict(merged_reaction_data_X_test)
        merged_reaction_data_accuracy = accuracy_score(merged_reaction_data_y_test, merged_reaction_pred)

        accuracyReportNoKp(key, main_feature_accuracy, merged_accuracy_data_accuracy, merged_speed_data_accuracy, merged_reaction_data_accuracy)
