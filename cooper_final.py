from __future__ import division
import pandas as pd;import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.svm import SVC;
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
data_util_file = './util/data_util.py'
import os
import sys
from data_util import *
import data_util as util
   
data = pd.read_csv('./data/bike_data.csv')

print(data.head())

col_names = data.columns.tolist()


del data['Casual Users']


def show_name(df):
	name = [x for x in globals() if globals()[x] is df][0]; 
	print("DataFrame Name is: %s" %name);

# ------------------------------------------------- #
# --- Section 3: Transformation and Cleaning up --- #
# ------------------------------------------------- #

#isolate our target data
target_result = data['Date']; 
#let's alter the values, so that everytime that we see:
#AECOM Employee, we set it to 1
#Everything else (contractor) set it to 0
y = np.where(target_result == 'AECOM Employee', 1, 0)

#transform using a label encoder
data = pd.get_dummies(data, columns=util.cat_features(data));

#if I had anything to drop, i'd specify it here
#but I get rid of this outside the working environment
to_drop =[]; 
feature_space = data.drop(to_drop, axis=1);

#remove feature space in case we need it later
features = feature_space.columns;  
X = feature_space.as_matrix().astype(np.float); 

#apply a scaler to the predictors
scaler = StandardScaler(); 
X = scaler.fit_transform(X);

#let's check to see if there's missing data
b = util.check_missing_data(data); 
if(b):
	print('Found Missing Data'); 
	show_name(data); 
	print('\n');
else:
	print('No Missing Data!');
	show_name(data); 
	print('\n');

#check to make sure that we've not done anything crazy at this point
print("Feature space contains %d records and %d columns" % X.shape); 
print("Number of Response Types:", np.unique(y));  


# ---------------------------------- #
# --- Section 4: Evaluate Models --- #
# ---------------------------------- #

print("Support Vector Machine:"); 
print("%.4f" % util.accuracy(y, util.run_cv(X,y,SVC))); 
print("Random Forest:"); 
print("%.4f" % util.accuracy(y, util.run_cv(X,y,RF))); 
print("K-Nearest-Neighbors:"); 
print("%.4f" % util.accuracy(y, util.run_cv(X,y,KNN))); 

# ------------------------------------- #
# --- Section 5: Confusion Matrices --- #
# ------------------------------------- #

y = np.array(y)
class_names = np.unique(y)
np.set_printoptions(precision=2)

confusion_matrix_SVC = confusion_matrix(y, util.run_cv(X,y,SVC)); 
confusion_matrix_RF = confusion_matrix(y, util.run_cv(X,y,RF)); 
confusion_matrix_KNN = confusion_matrix(y, util.run_cv(X,y,KNN)); 

plt.figure()
util.plot_confusion_matrix(confusion_matrix_SVC, classes=class_names,
                      title='Support Vector Machine, without normalization')
plt.figure()
util.plot_confusion_matrix(confusion_matrix_RF, classes=class_names,
                      title='Random Forest, without normalization')
plt.figure()
util.plot_confusion_matrix(confusion_matrix_KNN, classes=class_names,
                      title='K-Nearest-Neighbors, without normalization')

plt.figure()
util.plot_confusion_matrix(confusion_matrix_SVC, classes=class_names, normalize=True,
                      title='Support Vector Machine, with normalization')
plt.figure()
util.plot_confusion_matrix(confusion_matrix_RF, classes=class_names, normalize=True,
                      title='Random Forest, with normalization')
plt.figure()
util.plot_confusion_matrix(confusion_matrix_KNN, classes=class_names, normalize=True,
                      title='K-Nearest-Neighbors, with normalization')

#plt.show()


