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

DU_data = pd.read_csv('./csv_files/DU_task1.csv')

DU_X = DU_data.drop(['user','session', 'task', 'iteration'], axis=1)
DU_y= DU_data['user']

DU_X_train, DU_X_test, DU_y_train, DU_y_test = train_test_split(DU_X, DU_y, test_size = 0.4)

scaler = StandardScaler()
scaler.fit(DU_X_train)
DU_X_train = scaler.transform(DU_X_train)
DU_X_test = scaler.transform(DU_X_test)


scores = {}
scores_list = []
for k in range(1,15):
   model = KNeighborsClassifier(n_neighbors=k)
   model.fit(DU_X_train, DU_y_train)
   DU_y_pred = model.predict(DU_X_test)
   scores[k] = metrics.accuracy_score(DU_y_test,DU_y_pred)
   scores_list.append(metrics.accuracy_score(DU_y_test,DU_y_pred))
result = metrics.confusion_matrix(DU_y_test, DU_y_pred)
print("Confusion Matrix:")
print(result)
result1 = metrics.classification_report(DU_y_test, DU_y_pred)
print("Classification Report:",)
print (result1)

# sns.lineplot(range(1,30),scores_list, marker = "o")
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")

# plt.show()

selected_index = np.argmax(scores_list)
selected_k = range(1,30)[selected_index]

knn = KNeighborsClassifier(n_neighbors=selected_k)
knn.fit(DU_X_train, DU_y_train)
DU_y_pred = knn.predict(DU_X_test)

accuracy = accuracy_score(DU_y_test, DU_y_pred)
#precision = precision_score(DU_y_test, DU_y_pred, average="micro")
#recall = recall_score(DU_y_test, DU_y_pred, average="micro")

print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)