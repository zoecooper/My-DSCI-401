#Tried out the random forest classifier.

import pandas as pd
from sklearn import ensemble 
from sklearn.model_selection import train_test_split
from data_util import *
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import svm 
from sklearn import naive_bayes
from sklearn import ensemble 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Get a list of the categorical features for a given dataframe.
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe. 
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)

#Read in data.
data = pd.read_csv('./data/new_bike_data.csv')
val_data = pd.read_csv('./data/val_bike_data.csv')

#Set my response variable.
data_y = data['Holiday']

data_val_y = val_data['Holiday']

#Take out date- we do not need it.
del data['Date']
del val_data['Date']

print(data.head())

#Took out my dependent variable.
del data['Holiday']

del val_data['Holiday']

print(data.head())

#Transforming the data into a matrix of floats.  
x = data
val_x = val_data

#Use a label encoder.
x = pd.get_dummies(x, columns=cat_features(x))
val_x = pd.get_dummies(val_x, columns=cat_features(val_x))

print(x.head())
print(val_x.head())

X = x.as_matrix().astype(np.float) #making it matrix
val_X = val_x.as_matrix().astype(np.float) #making it matrix

#This is so there is no improper weighing in the matrix.
scaler = StandardScaler()
X = scaler.fit_transform(X)

val_scaler = StandardScaler()
val_X = val_scaler.fit_transform(val_X)

#Make sure everything looks right.
print("Data contains %d rows and %d columns" % X.shape)
print("Response Types:", np.unique(data_y))
print(X.shape)

#Make sure everything looks right.
print("Data contains %d rows and %d columns" % val_X.shape)
print("Response Types:", np.unique(data_val_y))
print(val_X.shape)

# Create training and test sets for later use.
x_train, x_test, y_train, y_test = train_test_split(X, data_y, test_size = 0.3, random_state = 4)

x_train_val, x_test_val, y_train_val, y_test_val = train_test_split(val_X, data_val_y, test_size = 0.3, random_state = 4)

# Build a sequence of models for different n_est and depth values. **NOTE: nest=10, depth=None is equivalent to the default.
n_est = [5, 10, 50, 100]
depth = [3, 6, None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test)
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_multiclass_classif_error_report(y_test, preds)
		
# Build a sequence of models for different n_est and depth values. **NOTE: nest=10, depth=None is equivalent to the default.
n_est = [5, 10, 50, 100]
depth = [3, 6, None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test_val)
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_multiclass_classif_error_report(y_test_val, preds)




