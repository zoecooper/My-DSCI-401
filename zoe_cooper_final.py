#can we predict if these riders are riding on a holiday

#Imported libraries.
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

#DATA TRANSFORMATIONS.

# Get a list of the categorical features.
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features. 
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)

#Out of python, I changed the original date column to just the months. 
#talk about metadata

#Read in data.
data = pd.read_csv('./data/new_bike_data.csv')

#print(data.head())

#Set my response variable.
data_y = data['Holiday']

#Take out date- we do not need it.
del data['Date']

print(data.head())

#Took out my dependent variable.
del data['Holiday']

print(data.head())
  
feature_space = data

#Use a label encoder.
feature_space = pd.get_dummies(feature_space, columns=cat_features(feature_space))

X = feature_space.as_matrix().astype(np.float) #making it matrix

#This is so there is no improper weighting in the matrix.
scaler = StandardScaler()
X = scaler.fit_transform(X)


#Make sure everything looks right.
print("Data contains %d rows and %d columns" % X.shape)
print("Response Types:", np.unique(data_y))
print(X.shape)

# Create training and test sets for later use.
x_train, x_test, y_train, y_test = train_test_split(X, data_y, test_size = 0.3, random_state = 4)

#MODEL OPTIMIZATION.
# Use a basic voting classifier with CV and Grid Search.
model1 = svm.SVC()
model2 = ensemble.RandomForestClassifier()
model3 = naive_bayes.GaussianNB()
voting_mod = ensemble.VotingClassifier(estimators=[('svm', model1), ('rf', model2), ('nb', model3)], voting='hard')

# Set up params for combined Grid Search on the voting model. Notice the convention for specifying 
# parameters foreach of the different models.
param_grid = {'svm__C':[0.2, 0.5, 1.0, 2.0, 5.0, 10.0], 'rf__n_estimators':[5, 10, 50, 100], 'rf__max_depth': [3, 6, None]}
best_voting_mod = GridSearchCV(estimator=voting_mod, param_grid=param_grid, cv=5)
best_voting_mod.fit(x_train, y_train)
print('Voting Ensemble Model Test Score: ' + str(best_voting_mod.score(x_test, y_test)))




