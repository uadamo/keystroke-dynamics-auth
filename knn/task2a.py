# Data Processing
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import seaborn as sns
import graphviz
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

DU_data = pd.read_csv('./csv_files/DU_task2a.csv')
DD_data = pd.read_csv('./csv_files/DD_task2a.csv')
UU_data = pd.read_csv('./csv_files/UU_task2a.csv')
UD_data = pd.read_csv('./csv_files/UD_task2a.csv')
trigraph_data = pd.read_csv('./csv_files/trigraph_task2a.csv')

accuracy_data = pd.read_csv('./csv_files/accuracy_task2a.csv')
speed_data = pd.read_csv('./csv_files/speed_task2a.csv')
reaction_data = pd.read_csv('./csv_files/reaction_task2a.csv')


merged_DU_accuracy_data = pd.merge(DU_data, accuracy_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_DU_speed_data = pd.merge(DU_data, speed_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 


merged_DU_reaction_data = pd.merge(UD_data,reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 


merged_UD_accuracy_data = pd.merge(UD_data, accuracy_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_UD_speed_data = pd.merge(UD_data, speed_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 



merged_UD_reaction_data = pd.merge(UD_data, reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_UU_accuracy_data = pd.merge(UU_data, accuracy_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_UU_speed_data = pd.merge(UU_data, speed_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 



merged_UU_reaction_data = pd.merge(UU_data, reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_DD_accuracy_data = pd.merge(DD_data, accuracy_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 

merged_DD_speed_data = pd.merge(DD_data, speed_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 



merged_DD_reaction_data = pd.merge(DD_data, reaction_data,  
                   on=['user','session', "task", "iteration"],  
                   how='inner') 



DU_X = DU_data.drop(['user','session', 'task', 'iteration'], axis=1)
DU_y= DU_data['user']

DU_X_train, DU_X_test, DU_y_train, DU_y_test = train_test_split(DU_X, DU_y, test_size = 0.4)

scaler_DU = StandardScaler()
scaler_DU.fit(DU_X_train)
DU_X_train = scaler_DU.transform(DU_X_train)
DU_X_test = scaler_DU.transform(DU_X_test)


scores_DU = {}
scores_DU_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(DU_X_train, DU_y_train)
   DU_y_pred = model.predict(DU_X_test)
   scores_DU[k] = metrics.accuracy_score(DU_y_test,DU_y_pred)
   scores_DU_list.append(metrics.accuracy_score(DU_y_test,DU_y_pred))
result_DU = metrics.confusion_matrix(DU_y_test, DU_y_pred)
# print("Confusion Matrix:")
# print(result_DU)
result_DU_1 = metrics.classification_report(DU_y_test, DU_y_pred)
# print("Classification Report:",)
# print (result_DU_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_DU = np.argmax(scores_DU_list)
selected_k_DU = range(1,30)[selected_index_DU]

knn_DU = KNeighborsClassifier(n_neighbors=selected_k_DU)
knn_DU.fit(DU_X_train, DU_y_train)
DU_y_pred = knn_DU.predict(DU_X_test)

DU_accuracy = accuracy_score(DU_y_test, DU_y_pred)
#precision = precision_score(DU_y_test, DU_y_pred, average="micro")
#recall = recall_score(DU_y_test, DU_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)


UD_X = UD_data.drop(['user','session', 'task', 'iteration'], axis=1)
UD_y= UD_data['user']

UD_X_train, UD_X_test, UD_y_train, UD_y_test = train_test_split(UD_X, UD_y, test_size = 0.4)

scaler_UD = StandardScaler()
scaler_UD.fit(UD_X_train)
UD_X_train = scaler_UD.transform(UD_X_train)
UD_X_test = scaler_UD.transform(UD_X_test)


scores_UD = {}
scores_UD_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(UD_X_train, UD_y_train)
   UD_y_pred = model.predict(UD_X_test)
   scores_UD[k] = metrics.accuracy_score(UD_y_test,UD_y_pred)
   scores_UD_list.append(metrics.accuracy_score(UD_y_test,UD_y_pred))
result_UD = metrics.confusion_matrix(UD_y_test, UD_y_pred)
# print("Confusion Matrix:")
# print(result_UD)
result_UD_1 = metrics.classification_report(UD_y_test, UD_y_pred)
# print("Classification Report:",)
# print (result_UD_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_UD = np.argmax(scores_UD_list)
selected_k_UD = range(1,30)[selected_index_UD]

knn_UD = KNeighborsClassifier(n_neighbors=selected_k_UD)
knn_UD.fit(UD_X_train, UD_y_train)
UD_y_pred = knn_UD.predict(UD_X_test)

UD_accuracy = accuracy_score(UD_y_test, UD_y_pred)
#precision = precision_score(UD_y_test, UD_y_pred, average="micro")
#recall = recall_score(UD_y_test, UD_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)


DD_X = DD_data.drop(['user','session', 'task', 'iteration'], axis=1)
DD_y= DD_data['user']

DD_X_train, DD_X_test, DD_y_train, DD_y_test = train_test_split(DD_X, DD_y, test_size = 0.4)

scaler_DD = StandardScaler()
scaler_DD.fit(DD_X_train)
DD_X_train = scaler_DD.transform(DD_X_train)
DD_X_test = scaler_DD.transform(DD_X_test)


scores_DD = {}
scores_DD_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(DD_X_train, DD_y_train)
   DD_y_pred = model.predict(DD_X_test)
   scores_DD[k] = metrics.accuracy_score(DD_y_test,DD_y_pred)
   scores_DD_list.append(metrics.accuracy_score(DD_y_test,DD_y_pred))
result_DD = metrics.confusion_matrix(DD_y_test, DD_y_pred)
# print("Confusion Matrix:")
# print(result_DD)
result_DD_1 = metrics.classification_report(DD_y_test, DD_y_pred)
# print("Classification Report:",)
# print (result_DD_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_DD = np.argmax(scores_DD_list)
selected_k_DD = range(1,30)[selected_index_DD]

knn_DD = KNeighborsClassifier(n_neighbors=selected_k_DD)
knn_DD.fit(DD_X_train, DD_y_train)
DD_y_pred = knn_DD.predict(DD_X_test)

DD_accuracy = accuracy_score(DD_y_test, DD_y_pred)
#precision = precision_score(DD_y_test, DD_y_pred, average="micro")
#recall = recall_score(DD_y_test, DD_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)


UU_X = UU_data.drop(['user','session', 'task', 'iteration'], axis=1)
UU_y= UU_data['user']

UU_X_train, UU_X_test, UU_y_train, UU_y_test = train_test_split(UU_X, UU_y, test_size = 0.4)

scaler_UU = StandardScaler()
scaler_UU.fit(UU_X_train)
UU_X_train = scaler_UU.transform(UU_X_train)
UU_X_test = scaler_UU.transform(UU_X_test)


scores_UU = {}
scores_UU_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(UU_X_train, UU_y_train)
   UU_y_pred = model.predict(UU_X_test)
   scores_UU[k] = metrics.accuracy_score(UU_y_test,UU_y_pred)
   scores_UU_list.append(metrics.accuracy_score(UU_y_test,UU_y_pred))
result_UU = metrics.confusion_matrix(UU_y_test, UU_y_pred)
# print("Confusion Matrix:")
# print(result_UU)
result_UU_1 = metrics.classification_report(UU_y_test, UU_y_pred)
# print("Classification Report:",)
# print (result_UU_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_UU = np.argmax(scores_UU_list)
selected_k_UU = range(1,30)[selected_index_UU]

knn_UU = KNeighborsClassifier(n_neighbors=selected_k_UU)
knn_UU.fit(UU_X_train, UU_y_train)
UU_y_pred = knn_UU.predict(UU_X_test)

UU_accuracy = accuracy_score(UU_y_test, UU_y_pred)
#precision = precision_score(UU_y_test, UU_y_pred, average="micro")
#recall = recall_score(UU_y_test, UU_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)



merged_DU_accuracy_X =merged_DU_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DU_accuracy_y=merged_DU_accuracy_data['user']

merged_DU_accuracy_X_train, merged_DU_accuracy_X_test, merged_DU_accuracy_y_train, merged_DU_accuracy_y_test = train_test_split(merged_DU_accuracy_X, merged_DU_accuracy_y, test_size = 0.4)

scaler_merged_DU_accuracy = StandardScaler()
scaler_merged_DU_accuracy.fit(merged_DU_accuracy_X_train)
merged_DU_accuracy_X_train = scaler_merged_DU_accuracy.transform(merged_DU_accuracy_X_train)
merged_DU_accuracy_X_test = scaler_merged_DU_accuracy.transform(merged_DU_accuracy_X_test)


scores_merged_DU_accuracy = {}
scores_merged_DU_accuracy_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_DU_accuracy_X_train, merged_DU_accuracy_y_train)
   merged_DU_accuracy_y_pred = model.predict(merged_DU_accuracy_X_test)
   scores_merged_DU_accuracy[k] = metrics.accuracy_score(merged_DU_accuracy_y_test,merged_DU_accuracy_y_pred)
   scores_merged_DU_accuracy_list.append(metrics.accuracy_score(merged_DU_accuracy_y_test,merged_DU_accuracy_y_pred))
result_merged_DU_accuracy = metrics.confusion_matrix(merged_DU_accuracy_y_test, merged_DU_accuracy_y_pred)
# print("Confusion Matrix:")
# print(result_merged_DU_accuracy)
result_merged_DU_accuracy_1 = metrics.classification_report(merged_DU_accuracy_y_test, merged_DU_accuracy_y_pred)
# print("Classification Report:",)
# print (result_merged_DU_accuracy_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_DU_accuracy = np.argmax(scores_merged_DU_accuracy_list)
selected_k_merged_DU_accuracy = range(1,30)[selected_index_merged_DU_accuracy]

knn_merged_DU_accuracy = KNeighborsClassifier(n_neighbors=selected_k_merged_DU_accuracy)
knn_merged_DU_accuracy.fit(merged_DU_accuracy_X_train, merged_DU_accuracy_y_train)
merged_DU_accuracy_y_pred = knn_merged_DU_accuracy.predict(merged_DU_accuracy_X_test)

merged_DU_accuracy_data_accuracy = accuracy_score(merged_DU_accuracy_y_test, merged_DU_accuracy_y_pred)
#precision = precision_score(merged_DU_accuracy_y_test, merged_DU_accuracy_y_pred, average="micro")
#recall = recall_score(merged_DU_accuracy_y_test, merged_DU_accuracy_y_pred, average="micro")


# print("Precision:", precision)
# print("Recall:", recall)


merged_DU_speed_X = merged_DU_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DU_speed_y= merged_DU_speed_data['user']

merged_DU_speed_X_train, merged_DU_speed_X_test, merged_DU_speed_y_train, merged_DU_speed_y_test = train_test_split(merged_DU_speed_X, merged_DU_speed_y, test_size = 0.4)

scaler_merged_DU_speed = StandardScaler()
scaler_merged_DU_speed.fit(merged_DU_speed_X_train)
merged_DU_speed_X_train = scaler_merged_DU_speed.transform(merged_DU_speed_X_train)
merged_DU_speed_X_test = scaler_merged_DU_speed.transform(merged_DU_speed_X_test)


scores_merged_DU_speed = {}
scores_merged_DU_speed_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_DU_speed_X_train, merged_DU_speed_y_train)
   merged_DU_speed_y_pred = model.predict(merged_DU_speed_X_test)
   scores_merged_DU_speed[k] = metrics.accuracy_score(merged_DU_speed_y_test,merged_DU_speed_y_pred)
   scores_merged_DU_speed_list.append(metrics.accuracy_score(merged_DU_speed_y_test,merged_DU_speed_y_pred))
result_merged_DU_speed = metrics.confusion_matrix(merged_DU_speed_y_test, merged_DU_speed_y_pred)
# print("Confusion Matrix:")
# print(result_merged_DU_speed)
result_merged_DU_speed_1 = metrics.classification_report(merged_DU_speed_y_test, merged_DU_speed_y_pred)
# print("Classification Report:",)
# print (result_merged_DU_speed_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_DU_speed = np.argmax(scores_merged_DU_speed_list)
selected_k_merged_DU_speed = range(1,30)[selected_index_merged_DU_speed]

knn_merged_DU_speed = KNeighborsClassifier(n_neighbors=selected_k_merged_DU_speed)
knn_merged_DU_speed.fit(merged_DU_speed_X_train, merged_DU_speed_y_train)
merged_DU_speed_y_pred = knn_merged_DU_speed.predict(merged_DU_speed_X_test)

merged_DU_speed_data_accuracy = accuracy_score(merged_DU_speed_y_test, merged_DU_speed_y_pred)
#precision = precision_score(merged_DU_speed_y_test, merged_DU_speed_y_pred, average="micro")
#recall = recall_score(merged_DU_speed_y_test, merged_DU_speed_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)





merged_DU_reaction_X = merged_DU_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DU_reaction_y= merged_DU_reaction_data['user']

merged_DU_reaction_X_train, merged_DU_reaction_X_test, merged_DU_reaction_y_train, merged_DU_reaction_y_test = train_test_split(merged_DU_reaction_X, merged_DU_reaction_y, test_size = 0.4)

scaler_merged_DU_reaction = StandardScaler()
scaler_merged_DU_reaction.fit(merged_DU_reaction_X_train)
merged_DU_reaction_X_train = scaler_merged_DU_reaction.transform(merged_DU_reaction_X_train)
merged_DU_reaction_X_test = scaler_merged_DU_reaction.transform(merged_DU_reaction_X_test)


scores_merged_DU_reaction = {}
scores_merged_DU_reaction_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_DU_reaction_X_train, merged_DU_reaction_y_train)
   merged_DU_reaction_y_pred = model.predict(merged_DU_reaction_X_test)
   scores_merged_DU_reaction[k] = metrics.accuracy_score(merged_DU_reaction_y_test,merged_DU_reaction_y_pred)
   scores_merged_DU_reaction_list.append(metrics.accuracy_score(merged_DU_reaction_y_test,merged_DU_reaction_y_pred))
result_merged_DU_reaction = metrics.confusion_matrix(merged_DU_reaction_y_test, merged_DU_reaction_y_pred)
# print("Confusion Matrix:")
# print(result_merged_DU_reaction)
result_merged_DU_reaction_1 = metrics.classification_report(merged_DU_reaction_y_test, merged_DU_reaction_y_pred)
# print("Classification Report:",)
# print (result_merged_DU_reaction_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_DU_reaction = np.argmax(scores_merged_DU_reaction_list)
selected_k_merged_DU_reaction = range(1,30)[selected_index_merged_DU_reaction]

knn_merged_DU_reaction = KNeighborsClassifier(n_neighbors=selected_k_merged_DU_reaction)
knn_merged_DU_reaction.fit(merged_DU_reaction_X_train, merged_DU_reaction_y_train)
merged_DU_reaction_y_pred = knn_merged_DU_reaction.predict(merged_DU_reaction_X_test)

merged_DU_reaction_data_accuracy = accuracy_score(merged_DU_reaction_y_test, merged_DU_reaction_y_pred)
#precision = precision_score(merged_DU_reaction_y_test, merged_DU_reaction_y_pred, average="micro")
#recall = recall_score(merged_DU_reaction_y_test, merged_DU_reaction_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)


merged_UD_accuracy_X =merged_UD_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UD_accuracy_y=merged_UD_accuracy_data['user']

merged_UD_accuracy_X_train, merged_UD_accuracy_X_test, merged_UD_accuracy_y_train, merged_UD_accuracy_y_test = train_test_split(merged_UD_accuracy_X, merged_UD_accuracy_y, test_size = 0.4)

scaler_merged_UD_accuracy = StandardScaler()
scaler_merged_UD_accuracy.fit(merged_UD_accuracy_X_train)
merged_UD_accuracy_X_train = scaler_merged_UD_accuracy.transform(merged_UD_accuracy_X_train)
merged_UD_accuracy_X_test = scaler_merged_UD_accuracy.transform(merged_UD_accuracy_X_test)


scores_merged_UD_accuracy = {}
scores_merged_UD_accuracy_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_UD_accuracy_X_train, merged_UD_accuracy_y_train)
   merged_UD_accuracy_y_pred = model.predict(merged_UD_accuracy_X_test)
   scores_merged_UD_accuracy[k] = metrics.accuracy_score(merged_UD_accuracy_y_test,merged_UD_accuracy_y_pred)
   scores_merged_UD_accuracy_list.append(metrics.accuracy_score(merged_UD_accuracy_y_test,merged_UD_accuracy_y_pred))
result_merged_UD_accuracy = metrics.confusion_matrix(merged_UD_accuracy_y_test, merged_UD_accuracy_y_pred)
# print("Confusion Matrix:")
# print(result_merged_UD_accuracy)
result_merged_UD_accuracy_1 = metrics.classification_report(merged_UD_accuracy_y_test, merged_UD_accuracy_y_pred)
# print("Classification Report:",)
# print (result_merged_UD_accuracy_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_UD_accuracy = np.argmax(scores_merged_UD_accuracy_list)
selected_k_merged_UD_accuracy = range(1,30)[selected_index_merged_UD_accuracy]

knn_merged_UD_accuracy = KNeighborsClassifier(n_neighbors=selected_k_merged_UD_accuracy)
knn_merged_UD_accuracy.fit(merged_UD_accuracy_X_train, merged_UD_accuracy_y_train)
merged_UD_accuracy_y_pred = knn_merged_UD_accuracy.predict(merged_UD_accuracy_X_test)

merged_UD_accuracy_data_accuracy = accuracy_score(merged_UD_accuracy_y_test, merged_UD_accuracy_y_pred)
#precision = precision_score(merged_UD_accuracy_y_test, merged_UD_accuracy_y_pred, average="micro")
#recall = recall_score(merged_UD_accuracy_y_test, merged_UD_accuracy_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)


merged_UD_speed_X = merged_UD_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UD_speed_y= merged_UD_speed_data['user']

merged_UD_speed_X_train, merged_UD_speed_X_test, merged_UD_speed_y_train, merged_UD_speed_y_test = train_test_split(merged_UD_speed_X, merged_UD_speed_y, test_size = 0.4)

scaler_merged_UD_speed = StandardScaler()
scaler_merged_UD_speed.fit(merged_UD_speed_X_train)
merged_UD_speed_X_train = scaler_merged_UD_speed.transform(merged_UD_speed_X_train)
merged_UD_speed_X_test = scaler_merged_UD_speed.transform(merged_UD_speed_X_test)


scores_merged_UD_speed = {}
scores_merged_UD_speed_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_UD_speed_X_train, merged_UD_speed_y_train)
   merged_UD_speed_y_pred = model.predict(merged_UD_speed_X_test)
   scores_merged_UD_speed[k] = metrics.accuracy_score(merged_UD_speed_y_test,merged_UD_speed_y_pred)
   scores_merged_UD_speed_list.append(metrics.accuracy_score(merged_UD_speed_y_test,merged_UD_speed_y_pred))
result_merged_UD_speed = metrics.confusion_matrix(merged_UD_speed_y_test, merged_UD_speed_y_pred)
# print("Confusion Matrix:")
# print(result_merged_UD_speed)
result_merged_UD_speed_1 = metrics.classification_report(merged_UD_speed_y_test, merged_UD_speed_y_pred)
# print("Classification Report:",)
# print (result_merged_UD_speed_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_UD_speed = np.argmax(scores_merged_UD_speed_list)
selected_k_merged_UD_speed = range(1,30)[selected_index_merged_UD_speed]

knn_merged_UD_speed = KNeighborsClassifier(n_neighbors=selected_k_merged_UD_speed)
knn_merged_UD_speed.fit(merged_UD_speed_X_train, merged_UD_speed_y_train)
merged_UD_speed_y_pred = knn_merged_UD_speed.predict(merged_UD_speed_X_test)

merged_UD_speed_data_accuracy = accuracy_score(merged_UD_speed_y_test, merged_UD_speed_y_pred)
#precision = precision_score(merged_UD_speed_y_test, merged_UD_speed_y_pred, average="micro")
#recall = recall_score(merged_UD_speed_y_test, merged_UD_speed_y_pred, average="micro")



# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()




merged_UD_reaction_X = merged_UD_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UD_reaction_y= merged_UD_reaction_data['user']

merged_UD_reaction_X_train, merged_UD_reaction_X_test, merged_UD_reaction_y_train, merged_UD_reaction_y_test = train_test_split(merged_UD_reaction_X, merged_UD_reaction_y, test_size = 0.4)

scaler_merged_UD_reaction = StandardScaler()
scaler_merged_UD_reaction.fit(merged_UD_reaction_X_train)
merged_UD_reaction_X_train = scaler_merged_UD_reaction.transform(merged_UD_reaction_X_train)
merged_UD_reaction_X_test = scaler_merged_UD_reaction.transform(merged_UD_reaction_X_test)


scores_merged_UD_reaction = {}
scores_merged_UD_reaction_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_UD_reaction_X_train, merged_UD_reaction_y_train)
   merged_UD_reaction_y_pred = model.predict(merged_UD_reaction_X_test)
   scores_merged_UD_reaction[k] = metrics.accuracy_score(merged_UD_reaction_y_test,merged_UD_reaction_y_pred)
   scores_merged_UD_reaction_list.append(metrics.accuracy_score(merged_UD_reaction_y_test,merged_UD_reaction_y_pred))
result_merged_UD_reaction = metrics.confusion_matrix(merged_UD_reaction_y_test, merged_UD_reaction_y_pred)
# print("Confusion Matrix:")
# print(result_merged_UD_reaction)
result_merged_UD_reaction_1 = metrics.classification_report(merged_UD_reaction_y_test, merged_UD_reaction_y_pred)
# print("Classification Report:",)
# print (result_merged_UD_reaction_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_UD_reaction = np.argmax(scores_merged_UD_reaction_list)
selected_k_merged_UD_reaction = range(1,30)[selected_index_merged_UD_reaction]

knn_merged_UD_reaction = KNeighborsClassifier(n_neighbors=selected_k_merged_UD_reaction)
knn_merged_UD_reaction.fit(merged_UD_reaction_X_train, merged_UD_reaction_y_train)
merged_UD_reaction_y_pred = knn_merged_UD_reaction.predict(merged_UD_reaction_X_test)

merged_UD_reaction_data_accuracy = accuracy_score(merged_UD_reaction_y_test, merged_UD_reaction_y_pred)
#precision = precision_score(merged_UD_reaction_y_test, merged_UD_reaction_y_pred, average="micro")
#recall = recall_score(merged_UD_reaction_y_test, merged_UD_reaction_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)

merged_DD_accuracy_X =merged_DD_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DD_accuracy_y=merged_DD_accuracy_data['user']

merged_DD_accuracy_X_train, merged_DD_accuracy_X_test, merged_DD_accuracy_y_train, merged_DD_accuracy_y_test = train_test_split(merged_DD_accuracy_X, merged_DD_accuracy_y, test_size = 0.4)

scaler_merged_DD_accuracy = StandardScaler()
scaler_merged_DD_accuracy.fit(merged_DD_accuracy_X_train)
merged_DD_accuracy_X_train = scaler_merged_DD_accuracy.transform(merged_DD_accuracy_X_train)
merged_DD_accuracy_X_test = scaler_merged_DD_accuracy.transform(merged_DD_accuracy_X_test)


scores_merged_DD_accuracy = {}
scores_merged_DD_accuracy_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_DD_accuracy_X_train, merged_DD_accuracy_y_train)
   merged_DD_accuracy_y_pred = model.predict(merged_DD_accuracy_X_test)
   scores_merged_DD_accuracy[k] = metrics.accuracy_score(merged_DD_accuracy_y_test,merged_DD_accuracy_y_pred)
   scores_merged_DD_accuracy_list.append(metrics.accuracy_score(merged_DD_accuracy_y_test,merged_DD_accuracy_y_pred))
result_merged_DD_accuracy = metrics.confusion_matrix(merged_DD_accuracy_y_test, merged_DD_accuracy_y_pred)
# print("Confusion Matrix:")
# print(result_merged_DD_accuracy)
result_merged_DD_accuracy_1 = metrics.classification_report(merged_DD_accuracy_y_test, merged_DD_accuracy_y_pred)
# print("Classification Report:",)
# print (result_merged_DD_accuracy_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_DD_accuracy = np.argmax(scores_merged_DD_accuracy_list)
selected_k_merged_DD_accuracy = range(1,30)[selected_index_merged_DD_accuracy]

knn_merged_DD_accuracy = KNeighborsClassifier(n_neighbors=selected_k_merged_DD_accuracy)
knn_merged_DD_accuracy.fit(merged_DD_accuracy_X_train, merged_DD_accuracy_y_train)
merged_DD_accuracy_y_pred = knn_merged_DD_accuracy.predict(merged_DD_accuracy_X_test)

merged_DD_accuracy_data_accuracy = accuracy_score(merged_DD_accuracy_y_test, merged_DD_accuracy_y_pred)
#precision = precision_score(merged_DD_accuracy_y_test, merged_DD_accuracy_y_pred, average="micro")
#recall = recall_score(merged_DD_accuracy_y_test, merged_DD_accuracy_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)


merged_DD_speed_X = merged_DD_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DD_speed_y= merged_DD_speed_data['user']

merged_DD_speed_X_train, merged_DD_speed_X_test, merged_DD_speed_y_train, merged_DD_speed_y_test = train_test_split(merged_DD_speed_X, merged_DD_speed_y, test_size = 0.4)

scaler_merged_DD_speed = StandardScaler()
scaler_merged_DD_speed.fit(merged_DD_speed_X_train)
merged_DD_speed_X_train = scaler_merged_DD_speed.transform(merged_DD_speed_X_train)
merged_DD_speed_X_test = scaler_merged_DD_speed.transform(merged_DD_speed_X_test)


scores_merged_DD_speed = {}
scores_merged_DD_speed_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_DD_speed_X_train, merged_DD_speed_y_train)
   merged_DD_speed_y_pred = model.predict(merged_DD_speed_X_test)
   scores_merged_DD_speed[k] = metrics.accuracy_score(merged_DD_speed_y_test,merged_DD_speed_y_pred)
   scores_merged_DD_speed_list.append(metrics.accuracy_score(merged_DD_speed_y_test,merged_DD_speed_y_pred))
result_merged_DD_speed = metrics.confusion_matrix(merged_DD_speed_y_test, merged_DD_speed_y_pred)
# print("Confusion Matrix:")
# print(result_merged_DD_speed)
result_merged_DD_speed_1 = metrics.classification_report(merged_DD_speed_y_test, merged_DD_speed_y_pred)
# print("Classification Report:",)
# print (result_merged_DD_speed_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_DD_speed = np.argmax(scores_merged_DD_speed_list)
selected_k_merged_DD_speed = range(1,30)[selected_index_merged_DD_speed]

knn_merged_DD_speed = KNeighborsClassifier(n_neighbors=selected_k_merged_DD_speed)
knn_merged_DD_speed.fit(merged_DD_speed_X_train, merged_DD_speed_y_train)
merged_DD_speed_y_pred = knn_merged_DD_speed.predict(merged_DD_speed_X_test)

merged_DD_speed_data_accuracy = accuracy_score(merged_DD_speed_y_test, merged_DD_speed_y_pred)
#precision = precision_score(merged_DD_speed_y_test, merged_DD_speed_y_pred, average="micro")
#recall = recall_score(merged_DD_speed_y_test, merged_DD_speed_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)


# print("Recall:", recall)


merged_DD_reaction_X = merged_DD_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_DD_reaction_y= merged_DD_reaction_data['user']

merged_DD_reaction_X_train, merged_DD_reaction_X_test, merged_DD_reaction_y_train, merged_DD_reaction_y_test = train_test_split(merged_DD_reaction_X, merged_DD_reaction_y, test_size = 0.4)

scaler_merged_DD_reaction = StandardScaler()
scaler_merged_DD_reaction.fit(merged_DD_reaction_X_train)
merged_DD_reaction_X_train = scaler_merged_DD_reaction.transform(merged_DD_reaction_X_train)
merged_DD_reaction_X_test = scaler_merged_DD_reaction.transform(merged_DD_reaction_X_test)


scores_merged_DD_reaction = {}
scores_merged_DD_reaction_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_DD_reaction_X_train, merged_DD_reaction_y_train)
   merged_DD_reaction_y_pred = model.predict(merged_DD_reaction_X_test)
   scores_merged_DD_reaction[k] = metrics.accuracy_score(merged_DD_reaction_y_test,merged_DD_reaction_y_pred)
   scores_merged_DD_reaction_list.append(metrics.accuracy_score(merged_DD_reaction_y_test,merged_DD_reaction_y_pred))
result_merged_DD_reaction = metrics.confusion_matrix(merged_DD_reaction_y_test, merged_DD_reaction_y_pred)
# print("Confusion Matrix:")
# print(result_merged_DD_reaction)
result_merged_DD_reaction_1 = metrics.classification_report(merged_DD_reaction_y_test, merged_DD_reaction_y_pred)
# print("Classification Report:",)
# print (result_merged_DD_reaction_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_DD_reaction = np.argmax(scores_merged_DD_reaction_list)
selected_k_merged_DD_reaction = range(1,30)[selected_index_merged_DD_reaction]

knn_merged_DD_reaction = KNeighborsClassifier(n_neighbors=selected_k_merged_DD_reaction)
knn_merged_DD_reaction.fit(merged_DD_reaction_X_train, merged_DD_reaction_y_train)
merged_DD_reaction_y_pred = knn_merged_DD_reaction.predict(merged_DD_reaction_X_test)

merged_DD_reaction_data_accuracy = accuracy_score(merged_DD_reaction_y_test, merged_DD_reaction_y_pred)
#precision = precision_score(merged_DD_reaction_y_test, merged_DD_reaction_y_pred, average="micro")
#recall = recall_score(merged_DD_reaction_y_test, merged_DD_reaction_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)

merged_UU_accuracy_X =merged_UU_accuracy_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UU_accuracy_y=merged_UU_accuracy_data['user']

merged_UU_accuracy_X_train, merged_UU_accuracy_X_test, merged_UU_accuracy_y_train, merged_UU_accuracy_y_test = train_test_split(merged_UU_accuracy_X, merged_UU_accuracy_y, test_size = 0.4)

scaler_merged_UU_accuracy = StandardScaler()
scaler_merged_UU_accuracy.fit(merged_UU_accuracy_X_train)
merged_UU_accuracy_X_train = scaler_merged_UU_accuracy.transform(merged_UU_accuracy_X_train)
merged_UU_accuracy_X_test = scaler_merged_UU_accuracy.transform(merged_UU_accuracy_X_test)


scores_merged_UU_accuracy = {}
scores_merged_UU_accuracy_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_UU_accuracy_X_train, merged_UU_accuracy_y_train)
   merged_UU_accuracy_y_pred = model.predict(merged_UU_accuracy_X_test)
   scores_merged_UU_accuracy[k] = metrics.accuracy_score(merged_UU_accuracy_y_test,merged_UU_accuracy_y_pred)
   scores_merged_UU_accuracy_list.append(metrics.accuracy_score(merged_UU_accuracy_y_test,merged_UU_accuracy_y_pred))
result_merged_UU_accuracy = metrics.confusion_matrix(merged_UU_accuracy_y_test, merged_UU_accuracy_y_pred)
# print("Confusion Matrix:")
# print(result_merged_UU_accuracy)
result_merged_UU_accuracy_1 = metrics.classification_report(merged_UU_accuracy_y_test, merged_UU_accuracy_y_pred)
# print("Classification Report:",)
# print (result_merged_UU_accuracy_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_UU_accuracy = np.argmax(scores_merged_UU_accuracy_list)
selected_k_merged_UU_accuracy = range(1,30)[selected_index_merged_UU_accuracy]

knn_merged_UU_accuracy = KNeighborsClassifier(n_neighbors=selected_k_merged_UU_accuracy)
knn_merged_UU_accuracy.fit(merged_UU_accuracy_X_train, merged_UU_accuracy_y_train)
merged_UU_accuracy_y_pred = knn_merged_UU_accuracy.predict(merged_UU_accuracy_X_test)

merged_UU_accuracy_data_accuracy = accuracy_score(merged_UU_accuracy_y_test, merged_UU_accuracy_y_pred)
#precision = precision_score(merged_UU_accuracy_y_test, merged_UU_accuracy_y_pred, average="micro")
#recall = recall_score(merged_UU_accuracy_y_test, merged_UU_accuracy_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)


merged_UU_speed_X = merged_UU_speed_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UU_speed_y= merged_UU_speed_data['user']

merged_UU_speed_X_train, merged_UU_speed_X_test, merged_UU_speed_y_train, merged_UU_speed_y_test = train_test_split(merged_UU_speed_X, merged_UU_speed_y, test_size = 0.4)

scaler_merged_UU_speed = StandardScaler()
scaler_merged_UU_speed.fit(merged_UU_speed_X_train)
merged_UU_speed_X_train = scaler_merged_UU_speed.transform(merged_UU_speed_X_train)
merged_UU_speed_X_test = scaler_merged_UU_speed.transform(merged_UU_speed_X_test)


scores_merged_UU_speed = {}
scores_merged_UU_speed_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_UU_speed_X_train, merged_UU_speed_y_train)
   merged_UU_speed_y_pred = model.predict(merged_UU_speed_X_test)
   scores_merged_UU_speed[k] = metrics.accuracy_score(merged_UU_speed_y_test,merged_UU_speed_y_pred)
   scores_merged_UU_speed_list.append(metrics.accuracy_score(merged_UU_speed_y_test,merged_UU_speed_y_pred))
result_merged_UU_speed = metrics.confusion_matrix(merged_UU_speed_y_test, merged_UU_speed_y_pred)
# print("Confusion Matrix:")
# print(result_merged_UU_speed)
result_merged_UU_speed_1 = metrics.classification_report(merged_UU_speed_y_test, merged_UU_speed_y_pred)
# print("Classification Report:",)
# print (result_merged_UU_speed_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_UU_speed = np.argmax(scores_merged_UU_speed_list)
selected_k_merged_UU_speed = range(1,30)[selected_index_merged_UU_speed]

knn_merged_UU_speed = KNeighborsClassifier(n_neighbors=selected_k_merged_UU_speed)
knn_merged_UU_speed.fit(merged_UU_speed_X_train, merged_UU_speed_y_train)
merged_UU_speed_y_pred = knn_merged_UU_speed.predict(merged_UU_speed_X_test)

merged_UU_speed_data_accuracy = accuracy_score(merged_UU_speed_y_test, merged_UU_speed_y_pred)
#precision = precision_score(merged_UU_speed_y_test, merged_UU_speed_y_pred, average="micro")
#recall = recall_score(merged_UU_speed_y_test, merged_UU_speed_y_pred, average="micro")

# print("Precision:", precision)
# print("Recall:", recall)

# print("Precision:", precision)
# print("Recall:", recall)


merged_UU_reaction_X = merged_UU_reaction_data.drop(['user','session', 'task', 'iteration'], axis=1)
merged_UU_reaction_y= merged_UU_reaction_data['user']

merged_UU_reaction_X_train, merged_UU_reaction_X_test, merged_UU_reaction_y_train, merged_UU_reaction_y_test = train_test_split(merged_UU_reaction_X, merged_UU_reaction_y, test_size = 0.4)

scaler_merged_UU_reaction = StandardScaler()
scaler_merged_UU_reaction.fit(merged_UU_reaction_X_train)
merged_UU_reaction_X_train = scaler_merged_UU_reaction.transform(merged_UU_reaction_X_train)
merged_UU_reaction_X_test = scaler_merged_UU_reaction.transform(merged_UU_reaction_X_test)


scores_merged_UU_reaction = {}
scores_merged_UU_reaction_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(merged_UU_reaction_X_train, merged_UU_reaction_y_train)
   merged_UU_reaction_y_pred = model.predict(merged_UU_reaction_X_test)
   scores_merged_UU_reaction[k] = metrics.accuracy_score(merged_UU_reaction_y_test,merged_UU_reaction_y_pred)
   scores_merged_UU_reaction_list.append(metrics.accuracy_score(merged_UU_reaction_y_test,merged_UU_reaction_y_pred))
result_merged_UU_reaction = metrics.confusion_matrix(merged_UU_reaction_y_test, merged_UU_reaction_y_pred)
# print("Confusion Matrix:")
# print(result_merged_UU_reaction)
result_merged_UU_reaction_1 = metrics.classification_report(merged_UU_reaction_y_test, merged_UU_reaction_y_pred)
# print("Classification Report:",)
# print (result_merged_UU_reaction_1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index_merged_UU_reaction = np.argmax(scores_merged_UU_reaction_list)
selected_k_merged_UU_reaction = range(1,30)[selected_index_merged_UU_reaction]

knn_merged_UU_reaction = KNeighborsClassifier(n_neighbors=selected_k_merged_UU_reaction)
knn_merged_UU_reaction.fit(merged_UU_reaction_X_train, merged_UU_reaction_y_train)
merged_UU_reaction_y_pred = knn_merged_UU_reaction.predict(merged_UU_reaction_X_test)

merged_UU_reaction_data_accuracy = accuracy_score(merged_UU_reaction_y_test, merged_UU_reaction_y_pred)
#precision = precision_score(merged_UU_reaction_y_test, merged_UU_reaction_y_pred, average="micro")
#recall = recall_score(merged_UU_reaction_y_test, merged_UU_reaction_y_pred, average="micro")

def accuracyReport(mainFeatureName, mainFeatureAccuracy, mergedAccuracy, mergedSpeed, mergedReaction):
    mergedMap = {"typing accuracy": mergedAccuracy, "typing speed": mergedSpeed,"reaction": mergedReaction }

    print(mainFeatureName + " accuracy :", mainFeatureAccuracy)
    for key, value in mergedMap.items():
        if(value - mainFeatureAccuracy >= 0.1):
            print(f"{mainFeatureName} merged with {key} accuracy: {value} STRONGLY IMPROVED original")
        elif(value - mainFeatureAccuracy < 0.1 and value - mainFeatureAccuracy > 0.05 and value > mainFeatureAccuracy):
            print(f"{mainFeatureName} merged with {key} accuracy: {value} improved original")
        else:
            print(f"{mainFeatureName} merged with {key} accuracy: {value}")
        
    print("\n")


accuracyReport("DU", DU_accuracy, merged_DU_accuracy_data_accuracy, merged_DU_speed_data_accuracy, merged_DU_reaction_data_accuracy)

accuracyReport("UD", UD_accuracy, merged_UD_accuracy_data_accuracy, merged_UD_speed_data_accuracy,  merged_UD_reaction_data_accuracy)

accuracyReport("UU", UU_accuracy, merged_UU_accuracy_data_accuracy, merged_UU_speed_data_accuracy, merged_UU_reaction_data_accuracy)

accuracyReport("DD", DD_accuracy, merged_DD_accuracy_data_accuracy, merged_DD_speed_data_accuracy,  merged_DD_reaction_data_accuracy)


# print("Recall:", recall)