# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

DU_data = pd.read_csv('./csv_files/random-forest/DU_.csv')
DD_data = pd.read_csv('./csv_files/random-forest/DD_.csv')
UU_data = pd.read_csv('./csv_files/random-forest/UU_.csv')
UD_data = pd.read_csv('./csv_files/random-forest/UD_.csv')
digraph_data = pd.read_csv('./csv_files/random-forest/digraph_.csv')
trigraph_data = pd.read_csv('./csv_files/random-forest/trigraph_.csv')


DU_data['user'] = DU_data['user'].map(lambda x: 1 if x == '1060d17b-2e21-4a6b-90f1-f451764a20d0' else 0)
DD_data['user'] = DD_data['user'].map(lambda x: 1 if x == '1060d17b-2e21-4a6b-90f1-f451764a20d0' else 0)
UU_data['user'] = UU_data['user'].map(lambda x: 1 if x == '1060d17b-2e21-4a6b-90f1-f451764a20d0' else 0)
UD_data['user'] = UD_data['user'].map(lambda x: 1 if x == '1060d17b-2e21-4a6b-90f1-f451764a20d0' else 0)
digraph_data['user'] = digraph_data['user'].map(lambda x: 1 if x == '1060d17b-2e21-4a6b-90f1-f451764a20d0' else 0)
trigraph_data['user'] = trigraph_data['user'].map(lambda x: 1 if x == '1060d17b-2e21-4a6b-90f1-f451764a20d0' else 0)

DU_nopred = DU_data.drop('user', axis=1)
DU_pred= DU_data['user']

DD_nopred = DD_data.drop('user', axis=1)
DD_pred= DD_data['user']

UU_nopred = UU_data.drop('user', axis=1)
UU_pred= UU_data['user']

UD_nopred = UD_data.drop('user', axis=1)
UD_pred= UD_data['user']

digraph_nopred = digraph_data.drop('user', axis=1)
digraph_pred= digraph_data['user']

trigraph_nopred = trigraph_data.drop('user', axis=1)
trigraph_pred= trigraph_data['user']

# Split the data into training and test sets
DU_nopred_train, DU_nopred_test, DU_pred_train, DU_pred_test = train_test_split(DU_nopred, DU_pred, test_size=0.2)

DD_nopred_train, DD_nopred_test, DD_pred_train, DD_pred_test = train_test_split(DD_nopred, DD_pred, test_size=0.2)

UU_nopred_train, UU_nopred_test, UU_pred_train, UU_pred_test = train_test_split(UU_nopred, UU_pred, test_size=0.2)

UD_nopred_train, UD_nopred_test, UD_pred_train, UD_pred_test = train_test_split(UD_nopred, UD_pred, test_size=0.2)

digraph_nopred_train, digraph_nopred_test, digraph_pred_train, digraph_pred_test = train_test_split(digraph_nopred, digraph_pred, test_size=0.2)

trigraph_nopred_train, trigraph_nopred_test, trigraph_pred_train, trigraph_pred_test = train_test_split(trigraph_nopred, trigraph_pred, test_size=0.2)

rf1 = RandomForestClassifier()
rf1.fit(DU_nopred_train, DU_pred_train)

rf2 = RandomForestClassifier()
rf2.fit(DD_nopred_train, DD_pred_train)

rf3 = RandomForestClassifier()
rf3.fit(UU_nopred_train, UU_pred_train)

rf4 = RandomForestClassifier()
rf4.fit(UD_nopred_train, UD_pred_train)

rf5 = RandomForestClassifier()
rf5.fit(digraph_nopred_train, digraph_pred_train)

rf6 = RandomForestClassifier()
rf6.fit(trigraph_nopred_train, trigraph_pred_train)


DU_prediction = rf1.predict(DU_nopred_test)
DU_accuracy = accuracy_score(DU_pred_test, DU_prediction)

DD_prediction = rf2.predict(DD_nopred_test)
DD_accuracy = accuracy_score(DD_pred_test, DD_prediction)

UU_prediction = rf3.predict(UU_nopred_test)
UU_accuracy = accuracy_score(UU_pred_test, UU_prediction)

UD_prediction = rf4.predict(UD_nopred_test)
UD_accuracy = accuracy_score(UD_pred_test, UD_prediction)

digraph_prediction = rf5.predict(digraph_nopred_test)
digraph_accuracy = accuracy_score(digraph_pred_test, digraph_prediction)

trigraph_prediction = rf6.predict(trigraph_nopred_test)
trigraph_accuracy = accuracy_score(trigraph_pred_test, trigraph_prediction)

print("DU Accuracy:", DU_accuracy)
print("UU Accuracy:", UU_accuracy)
print("DD Accuracy:", DD_accuracy)
print("UD Accuracy:", UD_accuracy)
print("digraph Accuracy:", digraph_accuracy)
print("trigraph Accuracy:", trigraph_accuracy)