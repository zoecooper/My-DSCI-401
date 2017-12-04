# First, I tried a SVM classifier on the churn data.

#Loaded in libraries. 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from data_util import *
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import ensemble 

#Import the data set.
data = pd.read_csv('./data/churn_data.csv')

#Look at it.
print(data.head())

# Got a list of the categorical features.
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Got the indices of the features.
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)

#Print that.
print(cat_feature_inds(data))

# Transformed the data to a "one-hot encoding".
data = pd.get_dummies(data, columns=cat_features(data))

#Look at the data again.
print(data.head())

# Select x and y data

features = list(data)
features.remove('CustID')
features.remove('Churn_Yes')
features.remove('Churn_No')
data_x = data[features]
data_y = data['Churn_Yes']

# Use label encoding to convert the different class labels to unique numbers.
le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)

# Split the training and test sets.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Make and look at 2 models: (1)Gini Impurity criteria and (2)Information Gain criteria.
print('----------- DTREE W GINI IMPURITY CRITERION ------------------')
dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(x_train, y_train)
preds_gini = dtree_gini_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_gini)

print('\n----------- DTREE W ENTROPY CRITERION -----------------------')
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(x_train, y_train)
preds_entropy = dtree_entropy_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_entropy)

# Next, tried the random forest classifier on the churn data.

#Loaded in libraries.
import pandas as pd
from sklearn import ensemble 
from sklearn.model_selection import train_test_split
from data_util import *

#Imported the data set.
data = pd.read_csv('./data/churn_data.csv')

#Displayed data.
print(data.head())

# Got a list of the categorical features.
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Got the indices of the categorical features.	
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)

print(cat_feature_inds(data))

# Transformed the df to a "one-hot encoding".
data = pd.get_dummies(data, columns=cat_features(data))

#Look at it again.
print(data.head())

# Got features and x and y (response) data.
features = list(data)
features.remove('CustID')
features.remove('Churn_Yes')
features.remove('Churn_No')
data_x = data[features]
data_y = data['Churn_Yes']

# Split training and test sets.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Built models for different n_est and depth values.
n_est = [5, 10, 50, 100]
depth = [3, 6, None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make preds.
		preds = mod.predict(x_test)
		print('---------- EVALUATE MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_multiclass_classif_error_report(y_test, preds)

# Last, tried a niave bayesian classifier (Gaussian) on the churn data set.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import naive_bayes
from data_util import *

data = pd.read_csv('./data/churn_data.csv')

print(data.head())

# Got a list of the categorical features.
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Got the indices of the features.
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
	
#Print that.
print(cat_feature_inds(data))

# Transformed the data to a "one-hot encoding".
data = pd.get_dummies(data, columns=cat_features(data))

#Look at the data again.
print(data.head())

# Select x and y data

features = list(data)
features.remove('CustID')
features.remove('Churn_Yes')
features.remove('Churn_No')
data_x = data[features]
data_y = data['Churn_Yes']

#Use label encoding to convert the different class labels to unique numbers.
le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)

# Split training and test sets.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build and evaluate the model
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds)

# Illustrate recoding numeric classes back into original (text-based) labels.
y_test_labs = le.inverse_transform(y_test)
pred_labs = le.inverse_transform(preds)
print('(Actual, Predicted): \n' + str(zip(y_test_labs, pred_labs)))

# I chose to use the random forest classifier on the churn validation data. 

og_churn = pd.read_csv('./data/churn_data.csv')

# Got a list of the categorical features. 
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return list(filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe)))

#Deleted Cust ID like before.
del og_churn['CustID']

# Selected x and y data
features = list(og_churn)
features.remove('Churn')

data_x = og_churn[features]
data_x = pd.get_dummies(data_x, columns=cat_features(data_x))
data_y = og_churn['Churn']

# "One - hot encoding"
le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)

#Did cross validations w/ validation data.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)
mod = ensemble.RandomForestClassifier(n_estimators=100, max_depth=6)
mod.fit(x_train, y_train)

val_churn = pd.read_csv('./data/churn_validation.csv')
del val_churn['CustID']
del val_churn['Churn']

val_churn = pd.get_dummies(val_churn, columns=cat_features(val_churn))
valxtrain, valxtest, valytrain, valytest = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)
# Made preds.
preds = mod.predict(valxtest)
print('Churn Validation Predictions as follows')
# Print results.
print_multiclass_classif_error_report(y_test, preds)



