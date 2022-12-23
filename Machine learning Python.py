##### Machine learning 
##export libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import os
import yellowbrick
import pickle

from matplotlib.collections import PathCollection
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split# Import train_test_split function
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC, ConfusionMatrix
from yellowbrick.style import set_palette
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.contrib.wrapper import wrap
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
from pdpbox import pdp


############## Prediction models
### prepare data set 
X=data.drop(columns='target')
y=data['target']

### normalizing
scaler=MinMaxScaler()
X=scaler.fit_transform(X)


### define the function for learning curve
def plot_LearningCurv(model):
    loglc = LearningCurve(model,  title='Logistic Regression Learning Curve')
    loglc.fit(X_train, y_train)
    loglc.finalize() 


### define the function for learning curve
def plot_RoC(model):
    logrocauc = ROCAUC(model, classes=['False', 'True'],
    title='Logistic Regression ROC AUC Plot')
    logrocauc.fit(X_train, y_train)
    logrocauc.score(X_test, y_test)
    logrocauc.finalize()
    plt.show()


### prepare train data and test data (randomly split)
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, shuffle=True)

##NOTE! NOTE! replace label name with [0,1]
### prediction models
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier(n_neighbors=4)
knn_model.fit(X_train,y_train)
y_knn_pred=knn_model.predict(X_test)
KNNAcc = accuracy_score(y_knn_pred, y_test)
print('.:. K-Nearest Neighbour Accuracy:'+'\033[1m {:.2f}%'.format(KNNAcc*100)+' .:.')
plot_RoC(knn_model)
plot_LearningCurv(knn_model)

# SVM
from sklearn.svm import SVC
svm_model=SVC(probability=True)
svm_model.fit(X_train,y_train)
y_svm_pred=svm_model.predict(X_test)
SVMacc = accuracy_score(y_svm_pred, y_test)
print('.:. SVM Neighbour Accuracy:'+'\033[1m {:.2f}%'.format(SVMacc*100)+' .:.')
plot_RoC(svm_model)
plot_LearningCurv(svm_model)

# Naive Bays
from sklearn.naive_bayes import GaussianNB
NB_model=GaussianNB(var_smoothing=0.08)
NB_model.fit(X_train, y_train)
y_pred_NB=NB_model.predict(X_test)
NBacc = accuracy_score(y_pred_NB, y_test)
print('.:. NB  Accuracy:'+'\033[1m {:.1f}%'.format(NBacc*100)+' .:.')
plot_RoC(NB_model)
plot_LearningCurv(NB_model)

# Random Classifier
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=20, min_samples_split=15)

RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)
RFacc = accuracy_score(y_pred_RF, y_test)
print('.:. RF  Accuracy:'+'\033[1m {:.1f}%'.format(RFacc*100)+' .:.')
plot_RoC(RF_model)
plot_LearningCurv(RF_model)

##-----------------------------------------end----------------------------------------## 