import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
housing = pd.read_csv("Boston-house-price-data.csv")
# print(housing.head())
#Used to show the overall view of the data and used to detect any unusual data
    # housing.hist(bins=50,figsize=(20,15))
    # plt.show()

#To give the info. of null values and datatypes of attributes
    # print(housing.info())

# SPLITING TEST AND TRAIN DATA
# 1)train_size - Used to seperate test and train data in the ratio.
# 2)random_state - Used to fix the random permutation points to allow our machine to see only the train data using (np.random.seed())
test_set,train_set=train_test_split(housing,train_size=0.2,random_state=42)

#Used to shuffle the data in a equal ratio to avoid Bias
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index , test_index in split.split(housing,housing['CHAS']):
    strat_test_set=housing.loc[test_index]
    strat_train_set=housing.loc[train_index]

#MISSING ATTRIBUTES
# 1.Get rid of the missing data points
# 2.Get rid of whole attribute
# 3.set missing values with (0/mean/medium)
A = housing.dropna(subset=["RM"]) #Option1
B = housing.drop('CHAS',axis='columns') #Option2
#Option 3 - To fill the missing values all over the dataset and newly added rows
imputer = SimpleImputer(strategy="median")
imputer.fit_transform(housing)

# LOOKING FOR CORRELATIONS
#It tells us how the attributes are co-related to each other and gives value in [-1,1]
corr_matrix=housing.corr()
# print(corr_matrix['MEDV'].sort_values(ascending=False))
#Lets see some plotting in interested +ve or -ve corr values
    # attributes=['MEDV','RM','ZN','LSTAT']
#Helps to detect the outliers of the corelated matrix
    # scatter_matrix(housing[attributes],figsize=(12,8))
    # plt.show()

#Feature scaling
"""This is used to scale the features values in to equal scalabile values
Like in this data if we see RM and MEDV are scalled in different ways so we need to match the scale for better accurecy
primarily 2 are imp.
1) normalization:
        (value-min)/(max-min)
        Sklearn provides a class called MinMaxScaler
2) Standardization:
        (value-mean)/std
        Sklearn provides a class called StandardScaler"""

housing=strat_train_set.drop('MEDV',axis=1)
housing_labels=strat_train_set["MEDV"].copy()

# CREATING A PIPELINE
#A pipeline is used to simply automate the things or code which can be further used in the project or other!
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    #add as many as needed ...
    ('std_scaler',StandardScaler())
])
housing_tr=my_pipeline.fit_transform(housing)

# Selecting our desired model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor #ensemble means assembling many classifiers to produce a regressor/Classifier
from sklearn.metrics import mean_squared_error
import numpy as np
# model = LinearRegression()
# model = DecisionTreeRegressor()
model= RandomForestRegressor()
model.fit(housing_tr,housing_labels)
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
# print(model.predict(prepared_data))
# print(list(some_labels))

#Evaluating the model
housing_predictions=model.predict(housing_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)
#Linear regression is not suitable because there is a greater mean squared error(mse)
#Dcision tree regressor has overfitted the training data so 0 mse
# print(rmse)

#Using better evaluation technique - Cross Validation
#Cross Validation= It divides the data into some groups and cross validates the data by traning some groups and testing it with other group
from sklearn.model_selection import cross_val_score
scores= cross_val_score(model,housing_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
#Here we see the error is not 0 but also less error comapared to LinearRegression

def print_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation:",scores.std())
#RandomForestRegressor gives the best output with minimum mse and std
# print_scores(rmse_scores)

#For saving the model and use it
# from joblib import dump,load
# dump(model,'Dragon.joblib')

#Testing out the model with test data
X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
# print(final_rmse)

#For using the model prepared by the engineer
from joblib import dump,load
model=load('Dragon.joblib')
input=np.array([[-0.44228927,-0.4898311,-1.37640684,-0.27288841,-0.34321545,0.36524574,
 -0.33092752 ,1.20235683 ,-1.0016859 , 0.05733231 ,-1.21003475 , 0.38110555,
 -0.57309194]])
a=model.predict(input)
print(a)