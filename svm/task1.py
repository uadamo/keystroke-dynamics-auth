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
from helpersSVM import featureAccuracyAnalysis;


DU_data = pd.read_csv('./csv_files/DU_task1.csv')
DD_data = pd.read_csv('./csv_files/DD_task1.csv')
UU_data = pd.read_csv('./csv_files/UU_task1.csv')
UD_data = pd.read_csv('./csv_files/UD_task1.csv')
trigraph_data = pd.read_csv('./csv_files/trigraph_task1.csv')

accuracy_data = pd.read_csv('./csv_files/accuracy_task1.csv')
keyPreference_data = pd.read_csv('./csv_files/keyPreference_task1.csv')
speed_data = pd.read_csv('./csv_files/speed_task1.csv')
reaction_data = pd.read_csv('./csv_files/reaction_task1.csv')

featureAccuracyAnalysis(DU_data, DD_data, UU_data, UD_data, trigraph_data, accuracy_data, speed_data, keyPreference_data, reaction_data)