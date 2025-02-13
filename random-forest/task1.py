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
import graphviz
from helpers import featureAccuracyAnalysis, featureAnalysisSingleData;

UD_data_all = pd.read_csv('./csv_files/UD_task1_all.csv')
DU_data = pd.read_csv('./csv_files/DU_task1.csv')
DD_data = pd.read_csv('./csv_files/DD_task1.csv')
UU_data = pd.read_csv('./csv_files/UU_task1.csv')
UD_data = pd.read_csv('./csv_files/UD_task1.csv')
trigraph_data = pd.read_csv('./csv_files/trigraph_task1.csv')

accuracy_data = pd.read_csv('./csv_files/accuracy_task1.csv')
keyPreference_data = pd.read_csv('./csv_files/keyPreference_task1.csv')
speed_data = pd.read_csv('./csv_files/speed_task1.csv')
reaction_data = pd.read_csv('./csv_files/reaction_task1.csv')


DU_accuracy_data = pd.merge(DU_data , accuracy_data, on=['user','session', "task", "iteration"], how='inner') 
DU_speed_data = pd.merge(DU_data, speed_data, on=['user','session', "task", "iteration"], how='inner') 
DU_keyPreference_data = pd.merge(DU_data, keyPreference_data,on=['user','session', "task", "iteration"], how='inner') 
DU_reaction_data = pd.merge(DU_data,reaction_data, on=['user','session', "task", "iteration"], how='inner')

DD_accuracy_data = pd.merge(DD_data , accuracy_data, on=['user','session', "task", "iteration"], how='inner') 
DD_speed_data = pd.merge(DD_data, speed_data, on=['user','session', "task", "iteration"], how='inner') 
DD_keyPreference_data = pd.merge(DD_data, keyPreference_data,on=['user','session', "task", "iteration"], how='inner') 
DD_reaction_data = pd.merge(DD_data,reaction_data, on=['user','session', "task", "iteration"], how='inner')

UU_accuracy_data = pd.merge(UU_data , accuracy_data, on=['user','session', "task", "iteration"], how='inner') 
UU_speed_data = pd.merge(UU_data, speed_data, on=['user','session', "task", "iteration"], how='inner') 
UU_keyPreference_data = pd.merge(UU_data, keyPreference_data,on=['user','session', "task", "iteration"], how='inner') 
UU_reaction_data = pd.merge(UU_data,reaction_data, on=['user','session', "task", "iteration"], how='inner')

UD_accuracy_data = pd.merge(UD_data , accuracy_data, on=['user','session', "task", "iteration"], how='inner') 
UD_speed_data = pd.merge(UD_data, speed_data, on=['user','session', "task", "iteration"], how='inner') 
UD_keyPreference_data = pd.merge(UD_data, keyPreference_data,on=['user','session', "task", "iteration"], how='inner') 
UD_reaction_data = pd.merge(UD_data,reaction_data, on=['user','session', "task", "iteration"], how='inner')


# accuracy, precision, recall, f1
#DU
results_DU = featureAnalysisSingleData(DU_data)
print(results_DU)

results_DU_accuracy = featureAnalysisSingleData(DU_accuracy_data)
print(results_DU_accuracy)
results_DU_speed = featureAnalysisSingleData(DU_speed_data)
print(results_DU_speed)

results_DU_keyPreference = featureAnalysisSingleData(DU_keyPreference_data)
print(results_DU_keyPreference)

results_DU_reaction = featureAnalysisSingleData(DU_reaction_data)
print(results_DU_reaction)
#DD

results_DD= featureAnalysisSingleData(DD_data)
print(results_DD)

results_DD_accuracy = featureAnalysisSingleData(DD_accuracy_data)
print(results_DD_accuracy)

results_DD_speed = featureAnalysisSingleData(DD_speed_data)
print(results_DD_speed)

results_DD_keyPreference = featureAnalysisSingleData(DD_keyPreference_data)
print(results_DD_keyPreference)

results_DD_reaction = featureAnalysisSingleData(DD_reaction_data)
print(results_DD_reaction)

# #UU

results_UU= featureAnalysisSingleData(UU_data)
print(results_UU)

results_UU_accuracy = featureAnalysisSingleData(UU_accuracy_data)
print(results_UU_accuracy)

results_UU_speed = featureAnalysisSingleData(UU_speed_data)
print(results_UU_speed)

results_UU_keyPreference = featureAnalysisSingleData(UU_keyPreference_data)
print(results_UU_keyPreference)

results_UU_reaction = featureAnalysisSingleData(UU_reaction_data)
print(results_UU_reaction)

# #UD

print("regular")
results_UD= featureAnalysisSingleData(UD_data)
print(results_UD)

print("all")
results_UD_all= featureAnalysisSingleData(UD_data_all)
print(results_UD_all)


results_UD_accuracy = featureAnalysisSingleData(UD_accuracy_data)
print(results_UD_accuracy)

results_UD_speed = featureAnalysisSingleData(UD_speed_data)
print(results_UD_speed)

results_UD_keyPreference = featureAnalysisSingleData(UD_keyPreference_data)
print(results_UD_keyPreference)


results_UD_reaction = featureAnalysisSingleData(UD_reaction_data)
print(results_UD_reaction)

