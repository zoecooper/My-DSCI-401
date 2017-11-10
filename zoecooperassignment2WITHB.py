import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV

#Read in data.
housing  = pd.read_csv('./data/AmesHousingSetB.csv')



#Viewed data.
housing.head()

#THIS DEMONSTRATES BUILDING THE BASELINE MODEL.
#Saw I needed to get categorical list for dataframe and get the indices.

def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))
    
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
    
print(cat_feature_inds(housing))


#Transformed the df.
housing = pd.get_dummies(housing, columns=cat_features(housing))

print(housing.head())

#Realized I needed to take out the N/As.
NEWhousing = housing.fillna(method='ffill')

#Redoing things after taking out the N/As.

def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
    

NEWHOUSING = pd.get_dummies(NEWhousing, columns=cat_features(NEWhousing))

#Realized price needed to be data y cause that is what we are predicted so created a list of features without it.
features = list(NEWHOUSING)
features.remove('SalePrice')

#Named x and y data.
data_x = NEWHOUSING[features]

data_y = NEWHOUSING['SalePrice']

#Split training and test sets.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
    
linear_mod = linear_model.LinearRegression()

#Fitted the model.

linear_mod.fit(x_train,y_train)

#Make predictions on test data and print.

preds = linear_mod.predict(x_test)

pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

print('MSE, MAE, R^2, EVS (Base Model): ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 
    
   
#THIS DEMONSTRATES USING LASSO REGRESSION.
#Made model and fit.
base_mod = linear_model.LinearRegression()

base_mod.fit(x_train,y_train)

#Made predictions.
preds = base_mod.predict(x_test)
print('R^2 (Base Model): ' + str(r2_score(y_test, preds)))

#Tried out different alphas.
alphas = [0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]

#Showed lasso regression fits for these above and used a for loop.
for a in alphas:
	lasso_mod = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	lasso_mod.fit(x_train, y_train)
	preds = lasso_mod.predict(x_test)
	print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(y_test, preds)))
      





