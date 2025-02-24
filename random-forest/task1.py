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
from helpers import featureAnalysis, featureAnalysisRecursiveCV, featureAnalysisSequentialSelector;
from functools import reduce

import xlwt 
from xlwt import Workbook 
  

DU_data = pd.read_csv('./csv_files/DU_task1.csv')
DD_data = pd.read_csv('./csv_files/DD_task1.csv')
UU_data = pd.read_csv('./csv_files/UU_task1.csv')
UD_data = pd.read_csv('./csv_files/UD_task1.csv')
digraph_data = pd.read_csv('./csv_files/digraph_task1.csv')
trigraph_data = pd.read_csv('./csv_files/trigraph_task1.csv')

DU_data_statistical = pd.read_csv('./csv_files/DU_statistical_task1.csv')
DD_data_statistical = pd.read_csv('./csv_files/DD_statistical_task1.csv')
UU_data_statistical = pd.read_csv('./csv_files/UU_statistical_task1.csv')
UD_data_statistical = pd.read_csv('./csv_files/UD_statistical_task1.csv')
digraph_data_statistical = pd.read_csv('./csv_files/digraph_statistical_task1.csv')
trigraph_data_statistical = pd.read_csv('./csv_files/trigraph_statistical_task1.csv')


accuracy_data = pd.read_csv('./csv_files/accuracy_task1.csv')
keyPreference_data = pd.read_csv('./csv_files/keyPreference_task1.csv')
speed_data = pd.read_csv('./csv_files/speed_task1.csv')
reaction_data = pd.read_csv('./csv_files/reaction_task1.csv')
UD_negative_data = pd.read_csv('./csv_files/UD_negative_task1.csv')
DU_negative_data = pd.read_csv('./csv_files/DU_negative_task1.csv')

non_temporal_features = [accuracy_data, keyPreference_data, speed_data, reaction_data, UD_negative_data, DU_negative_data]
merged_non_temporal_features = reduce(lambda left, right: pd.merge(left, right, on=['user','session', "task", "iteration"], how='inner'), non_temporal_features)


DU_non_temporal_data = pd.merge(DU_data, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 
DD_non_temporal_data = pd.merge(DD_data, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 
UU_non_temporal_data = pd.merge(UU_data, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 
UD_non_temporal_data = pd.merge(UD_data, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 
digraph_non_temporal_data = pd.merge(digraph_data, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 
trigraph_non_temporal_data = pd.merge(trigraph_data, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 


DU_non_temporal_and_statistic_data = pd.merge(DU_data_statistical, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 
DD_non_temporal_and_statistic_data = pd.merge(DD_data_statistical, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 
UU_non_temporal_and_statistic_data = pd.merge(UU_data_statistical, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 
UD_non_temporal_and_statistic_data = pd.merge(UD_data_statistical, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 
digraph_non_temporal_and_statistic_data =  pd.merge(digraph_data_statistical, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 
trigraph_non_temporal_and_statistic_data =  pd.merge(trigraph_data_statistical, merged_non_temporal_features, on=['user','session', "task", "iteration"], how='inner') 


def xlReportNoSelection():
    wb = xlwt.Workbook() 

    # single features - no selection
    ws_single = wb.add_sheet('task1-single-rf')
    ws_single.write(0,1,"accuracy")
    ws_single.write(0,2,"precision")
    ws_single.write(0,3,"recall")
    ws_single.write(0,4,"f1-score")

    pure_temporal = [featureAnalysis(DU_data),featureAnalysis(DD_data),featureAnalysis(UD_data),featureAnalysis(UU_data), featureAnalysis(digraph_data),featureAnalysis(trigraph_data)]
    
    ws_single.write(1,0,"DU")
    ws_single.write(2,0,"DD")
    ws_single.write(3,0,"UD")
    ws_single.write(4,0,"UU")
    ws_single.write(5,0,"digraph")
    ws_single.write(6,0,"trigraph")

    for i, values in enumerate(pure_temporal):
        ws_single.write(i+1,1,values[0])
        ws_single.write(i+1,2,values[1])
        ws_single.write(i+1,3,values[2])
        ws_single.write(i+1,4,values[3])
        
    # single+statistical - no selection
    ws_stats = wb.add_sheet('task1-statistical-rf')
    ws_stats.write(0,1,"accuracy")
    ws_stats.write(0,2,"precision")
    ws_stats.write(0,3,"recall")
    ws_stats.write(0,4,"f1-score")
    

    statistical = [featureAnalysis(DU_data_statistical),featureAnalysis(DD_data_statistical),featureAnalysis(UD_data_statistical),featureAnalysis(UU_data_statistical),featureAnalysis(digraph_data_statistical),featureAnalysis(trigraph_data_statistical)]
    
    ws_stats.write(1,0,"DU")
    ws_stats.write(2,0,"DD")
    ws_stats.write(3,0,"UD")
    ws_stats.write(4,0,"UU")
    ws_stats.write(5,0,"digraph")
    ws_stats.write(6,0,"trigraph")

    for i, values in enumerate(statistical):
        ws_stats.write(i+1,1,values[0])
        ws_stats.write(i+1,2,values[1])
        ws_stats.write(i+1,3,values[2])
        ws_stats.write(i+1,4,values[3])
        
    # non temporal - no filtering

    ws_non_temp = wb.add_sheet('task1-non-temporal-rf')
    ws_non_temp.write(0,1,"accuracy")
    ws_non_temp.write(0,2,"precision")
    ws_non_temp.write(0,3,"recall")
    ws_non_temp.write(0,4,"f1-score")
    
    non_temp = [featureAnalysis(DU_non_temporal_data),featureAnalysis(DD_non_temporal_data),featureAnalysis(UD_non_temporal_data),featureAnalysis(UU_non_temporal_data),featureAnalysis(digraph_non_temporal_data),featureAnalysis(trigraph_non_temporal_data)]
    
    ws_non_temp.write(1,0,"DU")
    ws_non_temp.write(2,0,"DD")
    ws_non_temp.write(3,0,"UD")
    ws_non_temp.write(4,0,"UU")
    ws_non_temp.write(5,0,"digraph")
    ws_non_temp.write(6,0,"trigraph")

    for i, values in enumerate(non_temp):
        ws_non_temp.write(i+1,1,values[0])
        ws_non_temp.write(i+1,2,values[1])
        ws_non_temp.write(i+1,3,values[2])
        ws_non_temp.write(i+1,4,values[3])
    
    # all

    ws_all = wb.add_sheet('task1-all-rf')
    ws_all.write(0,1,"accuracy")
    ws_all.write(0,2,"precision")
    ws_all.write(0,3,"recall")
    ws_all.write(0,4,"f1-score")
    
    all = [featureAnalysis(DU_non_temporal_and_statistic_data),featureAnalysis(DD_non_temporal_and_statistic_data),featureAnalysis(UD_non_temporal_and_statistic_data),featureAnalysis(UU_non_temporal_and_statistic_data),featureAnalysis(digraph_non_temporal_and_statistic_data),featureAnalysis(trigraph_non_temporal_and_statistic_data)]
    
    ws_all.write(1,0,"DU")
    ws_all.write(2,0,"DD")
    ws_all.write(3,0,"UD")
    ws_all.write(4,0,"UU")
    ws_all.write(5,0,"digraph")
    ws_all.write(6,0,"trigraph")

    for i, values in enumerate(all):
        ws_all.write(i+1,1,values[0])
        ws_all.write(i+1,2,values[1])
        ws_all.write(i+1,3,values[2])
        ws_all.write(i+1,4,values[3])
    
    wb.save("task1_rf_no_selection.xls")  

# xlReportNoSelection()


def xlReportRecursiveCVSelection():
    wb = xlwt.Workbook() 

    # single features - no selection
    ws = wb.add_sheet('task1-single-rf-recursive-CV')

    ws.write(0,1, "nr. of best features")
    ws.write(0,2, "removed features (least to most important)")
    ws.write(0,3,"accuracy (before filtering)")
    ws.write(0,4,"accuracy")
    ws.write(0,5,"precision")
    ws.write(0,6,"recall")
    ws.write(0,7,"f1-score")

    # featureAnalysisRecursiveCV(UD_non_temporal_and_statistic_data)

    feat = [featureAnalysisRecursiveCV(UD_non_temporal_and_statistic_data)]

    for i, values in enumerate(feat):
        ws.write(i+1,1,values[0])
        ws.write(i+1,2,values[1])
        ws.write(i+1,3,values[2])
        ws.write(i+1,4,values[3])
        ws.write(i+1,5,values[4])
        ws.write(i+1,6,values[5])
        ws.write(i+1,7,values[6])
    
    wb.save("task1_rf_recursive-CV.xls")  

# xlReportRecursiveCVSelection()


def xlReportSequentialSelection():
    wb = xlwt.Workbook() 

    # single features - no selection
    ws = wb.add_sheet('task1-rf-sequential')

    ws.write(0,1, "nr. of best features")
    ws.write(0,2, "removed features")
    ws.write(0,3,"accuracy (before filtering)")
    ws.write(0,4,"accuracy")
    ws.write(0,5,"precision")
    ws.write(0,6,"recall")
    ws.write(0,7,"f1-score")

    # featureAnalysisRecursiveCV(UD_non_temporal_and_statistic_data)

    feat = [featureAnalysisSequentialSelector(UD_non_temporal_and_statistic_data, 10), featureAnalysisSequentialSelector(UD_non_temporal_and_statistic_data, 12), featureAnalysisSequentialSelector(UD_non_temporal_and_statistic_data, 14), featureAnalysisSequentialSelector(UD_non_temporal_and_statistic_data, 16),  featureAnalysisSequentialSelector(UD_non_temporal_and_statistic_data, 18)]

    for i, values in enumerate(feat):
        ws.write(i+1,1,values[0])
        ws.write(i+1,2,values[1])
        ws.write(i+1,3,values[2])
        ws.write(i+1,4,values[3])
        ws.write(i+1,5,values[4])
        ws.write(i+1,6,values[5])
        ws.write(i+1,7,values[6])
    
    wb.save("task1_rf_sequential.xls")  

xlReportSequentialSelection()