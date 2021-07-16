# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:45:43 2021

@author: ktfin
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import plot_confusion_matrix
#%% Data Import and Trimming

file = r'ProcessParameterDataset.xlsx'
data = pd.read_excel(file)
data = data.replace(to_replace='non conductive line ', value=0)
trainData = pd.read_excel(file, usecols="A:C")

NuzzelData     = np.array(data['Nuzzel Speed (mm/min)'], dtype=np.float64)
FlowrateData   = np.array(data['Flowrate (ul/min)'], dtype=np.float64)
VoltageData    = np.array(data['Applied voltage (kV)'], dtype=np.float64)
ResistanceData = np.array(data['Resistance (kOhms)'], dtype=np.float64)
CombinedData   = np.array(trainData)

TrimmedData = np.delete(ResistanceData, np.where(ResistanceData == 0))
SortedData = np.sort(TrimmedData)
ResistanceMedian = np.median(TrimmedData)
HiLowData = np.zeros(np.size(ResistanceData))

for i in range(0, np.size(HiLowData)):
    if ResistanceData[i] > 0 and ResistanceData[i] <= ResistanceMedian:
        HiLowData[i] = 1
    elif ResistanceData[i] == 0 or ResistanceData[i] > ResistanceMedian:
        HiLowData[i] = 0

#%% Random Forest Classifier

X = CombinedData
y = HiLowData

X, y = make_classification(n_samples=105, n_features= 3, n_informative=3, n_redundant=0, random_state=0, shuffle=False)

RFC         = RandomForestClassifier(n_estimators=100, max_depth=3, max_features=None, n_jobs=-1)
RFC.fit(X, y)

rfcScore    = RFC.score(X,y)
rfcPredict  = RFC.predict(X)

print('Random Forest Score: ', rfcScore)
#print(rfcPredict)

#%% Confusion Matrix

confusion_matrix(y, rfcPredict)
print(confusion_matrix(y, rfcPredict))
recall = recall_score(y, rfcPredict)
prec = precision_score(y, rfcPredict)
f1 = 2 * (prec*recall)/(prec+recall)
fig, ax = plt.subplots(dpi=200)
plot_confusion_matrix(RFC, X, y, cmap= 'bone', ax=ax)
plt.title('Confusion Matrix: Random Forest Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
print('Recall Score:', recall)
print('Precision Score', prec)
print('F1 Score', f1)

#%% Linear Regression

LR = LinearRegression()
LR.fit(X,y)
LR_score = LR.score(X,y)


print('Linear Regression Score: ', LR_score)



#%% Importance

importance = RFC.feature_importances_

std = np.std([tree.feature_importances_ for tree in RFC.estimators_],axis=0)
indices = np.argsort(importance)[::-1]
label = []
for f in range(X.shape[1]):
    if indices[f] == 0:
        label.append('Nozzle Speed')
    elif indices[f] == 1:
        label.append('Flow Rate')
    elif indices[f] == 2:
        label.append('Applied Voltage')

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d.  %s (%f)" % (f + 1, label[f], importance[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure('feature importance', dpi=200)
plt.title("Feature Importance: Random Forest Classifier")
plt.bar(range(X.shape[1]), importance[indices],
        color=["dodgerblue", "goldenrod", "firebrick"], yerr=std[indices], align="center", width=0.7)
plt.xticks(range(X.shape[1]), label)
plt.xlim([-.5, X.shape[1]-.5])
plt.ylim([0,1])


#%% K-Fold Validation

k_fold = KFold(n_splits=5)
k_fold.get_n_splits(X)
# print(k_fold)

for train_index, test_index in k_fold.split(X,y):
      # print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      #print(X_train, X_test, y_train, y_test)
k_fold_score_RFC = cross_val_score(RFC, X, y, cv=k_fold, n_jobs=-1)
k_fold_score_LR = cross_val_score(LR, X, y, cv=None)
print('\n5-Fold Validation Score (RFC): ', k_fold_score_RFC)
print('5-Fold Validation Score (LR): ', k_fold_score_LR)





# Median of the resistance data. above 0 below median is good. everything else is bad


# confusion matrix 

# test against linear regression

# Define the meteric 