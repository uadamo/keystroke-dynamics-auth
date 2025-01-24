# Data Processing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import seaborn as sns
import graphviz
from helpersKNN import featureAccuracyAnalysisNoKp;


DU_data = pd.read_csv('./csv_files/DU_task2a.csv')
DD_data = pd.read_csv('./csv_files/DD_task2a.csv')
UU_data = pd.read_csv('./csv_files/UU_task2a.csv')
UD_data = pd.read_csv('./csv_files/UD_task2a.csv')
trigraph_data = pd.read_csv('./csv_files/trigraph_task2a.csv')

accuracy_data = pd.read_csv('./csv_files/accuracy_task2a.csv')
speed_data = pd.read_csv('./csv_files/speed_task2a.csv')
reaction_data = pd.read_csv('./csv_files/reaction_task2a.csv')

featureAccuracyAnalysisNoKp(DU_data, DD_data, UU_data, UD_data, trigraph_data, accuracy_data, speed_data, reaction_data)