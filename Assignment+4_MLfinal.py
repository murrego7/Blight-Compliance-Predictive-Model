
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     readonly/train.csv - the training set (all tickets issued 2004-2011)
#     readonly/test.csv - the test set (all tickets issued 2012-2016)
#     readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `readonly/train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `readonly/test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[56]:

#import pandas as pd
#import numpy as np

train = pd.read_csv('train.csv',encoding='ISO-8859-1')

#! cat readonly/test.csv > test.csv
test = pd.read_csv('test.csv', encoding='cp1252')

#! cat readonly/latlons.csv > latlons.csv
latlon = pd.read_csv('latlons.csv')

#! cat readonly/addresses.csv > addresses.csv
address = pd.read_csv('addresses.csv')

#Check dataframes's initial columns and shapes
print(train.columns)
print(test.columns)
print(latlon.columns)
print(address.columns)
print(train.shape)
print(test.shape)


# In[43]:

#Drop null-values compliance (NaN)
train = train.dropna(subset=['compliance'])

#Drop all columns with NaN
train = train.dropna(axis = 1,how  = 'all')
print(train.shape)

#Calculate total and percent missing data per features
percentagenull = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([percentagenull], axis=1, keys = ['Percent'])
missing_data.head(5)


# In[44]:

#Get rid of columns with >50% missing data
train.drop(missing_data[missing_data['Percent']>0.5].index,axis = 1,inplace = True)


# In[45]:

#Delete columns which have all values same
length = len(train.columns)
for i in range (length):
    if len(train[train.columns[i]].unique()) == 1:
         print(train.columns[i])
            
train.drop(labels = 'clean_up_cost' ,axis = 1,inplace = True)
test.drop(labels = 'clean_up_cost' ,axis = 1,inplace = True)
train.drop(labels = 'state_fee' ,axis = 1,inplace = True)
test.drop(labels = 'state_fee' ,axis = 1,inplace = True)
train.drop(labels = 'admin_fee' ,axis = 1,inplace = True)
test.drop(labels = 'admin_fee' ,axis = 1,inplace = True)


# In[46]:

#Merge train/test and (address + latlon) dataframes
train = pd.merge(train, pd.merge(latlon, address, how='inner', on = 'address'), on = 'ticket_id')
test = pd.merge(test, pd.merge(latlon, address, how='inner', on = 'address'), on = 'ticket_id')
print(train.shape)
print(test.shape)


# In[47]:

# Let's check again new dataframe's features
print(train.columns)
print(test.columns)


# In[48]:

# Remove useless variables from both train and test dataframes
remove = ['balance_due', 'payment_status', 'compliance_detail', 'agency_name', 'inspector_name', 
'violator_name', 'violation_street_name', 'mailing_address_str_name', 'city', 'violation_street_number',
 'state', 'zip_code', 'country',  'ticket_issued_date', 'hearing_date', 'violation_description',
 'discount_amount', 'payment_amount', 'disposition', 'address', 'violation_code', 'mailing_address_str_number']

removetest = ['non_us_str_code', 'agency_name', 'inspector_name', 'violator_name', 'violation_street_name', 
              'mailing_address_str_name', 'city', 'violation_street_number','state', 'zip_code', 'country', 
              'ticket_issued_date', 'hearing_date', 'violation_description', 'discount_amount',
             'grafitti_status', 'violation_zip_code', 'disposition', 'address', 'violation_code', 'mailing_address_str_number']
train.drop(remove, axis = 1, inplace = True)
test.drop(removetest, axis = 1, inplace = True)
print(train.shape)
print(test.shape)


# In[49]:

## Fill lat and long missing values with mean of the feature 
test['lat'] = test['lat'].fillna(test['lat'].mean())
train['lat'] = train['lat'].fillna(train['lat'].mean())

test['lon'] = test['lon'].fillna(test['lon'].mean())
train['lon'] = train['lon'].fillna(train['lon'].mean())

#print(train.isnull().sum())
#print(test.isnull().sum())


# In[50]:

#Set target value
y_train = train['compliance']
# Drop target value from training dataframe
X_train = train.drop(['compliance'], axis=1)
#Dataframe to test the model
X_test = test


# In[51]:

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0)

#Normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[18]:

from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

#Let's fit a Dummy model in order to make future comparisons
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
y_majority_predicted = dummy_majority.predict(X_test)
print('Accuracy of Dummy classifier on training set: {:.2f}'
     .format(dummy_majority.score(X_train_scaled, y_train)))
print('Accuracy of Dummy classifier on test set: {:.2f}'
     .format(dummy_majority.score(X_test_scaled, y_test)))
print('ROC AUC score Dummy classifier on test set: {:.2f}'
     .format(roc_auc_score(y_test, y_majority_predicted)))


# In[21]:

from sklearn.neighbors import KNeighborsClassifier

# Running K-nearest neighbors model
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_scaled, y_train)
ypred = knn.predict(X_test)

print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, y_test)))
print('ROC AUC score K-NN classifier on test set: {:.2f}'
     .format(roc_auc_score(y_test, ypred)))


# In[20]:

from sklearn.svm import LinearSVC
clf = LinearSVC().fit(X_train, y_train)
ypred = clf.predict(X_test)

# Running Support Vector Machines Model
print('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
print('ROC AUC score Linear SVC classifier on test set: {:.2f}'
     .format(roc_auc_score(y_test, ypred)))


# In[37]:

from sklearn.neural_network import MLPClassifier

#Running Neural Network Model 
clf = MLPClassifier(hidden_layer_sizes = [10, 10], alpha = 0.01,
                   random_state = 0, solver = 'lbfgs').fit(X_train_scaled, y_train)
ypred = clf.predict(X_test)
print('Accuracy of NN classifier on training set: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))
print('ROC AUC score NN classifier on test set: {:.2f}'
     .format(roc_auc_score(y_test, ypred)))


# In[29]:

from sklearn.ensemble import RandomForestRegressor

#Running Random Forest Model 
reg = RandomForestRegressor(max_depth = 10,random_state=0).fit(X_train, y_train)
ypred = reg.predict(X_test)
print('Accuracy of RF classifier on training set: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))
print('ROC AUC score of RF classifier on test set: {:.2f}'
     .format(roc_auc_score(y_test, ypred)))
#WE GOT THE BEST ROC AUC SCORE! 


# In[52]:

from sklearn.model_selection import  GridSearchCV

#Grid Search for finding Optimal Values
reg = RandomForestRegressor(max_depth = 10,random_state=0).fit(X_train, y_train)
grid_values = {'n_estimators': [10, 100], 'max_depth': [None, 30]}
grid_clf_auc = GridSearchCV(reg, param_grid=grid_values, scoring='roc_auc')
grid_clf_auc.fit(X_train, y_train)
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)


# In[53]:

## (Random Forest results)
df = pd.DataFrame(grid_clf_auc.predict(test), test.ticket_id)


# In[2]:

#def blight_model():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score


    train = pd.read_csv('train.csv',encoding='ISO-8859-1')
    test = pd.read_csv('test.csv', encoding='cp1252')
    latlon = pd.read_csv('latlons.csv')
    address = pd.read_csv('addresses.csv')
    
    for i in range(len(train.columns)): 
        if len(train[train.columns[i]].unique()) < 250:
            train[train.columns[i]] = train[train.columns[i]] .astype('category')
        
    train = train.dropna(subset=['compliance'])
    train = train.dropna(axis = 1,how  = 'all')

    percentagenull = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)
    missing_data = pd.concat([percentagenull], axis=1, keys = ['Percent'])
    
    train.drop(missing_data[missing_data['Percent']>0.5].index,axis = 1,inplace = True)
            
    train.drop(labels = 'clean_up_cost' ,axis = 1,inplace = True)
    test.drop(labels = 'clean_up_cost' ,axis = 1,inplace = True)
    train.drop(labels = 'state_fee' ,axis = 1,inplace = True)
    test.drop(labels = 'state_fee' ,axis = 1,inplace = True)
    train.drop(labels = 'admin_fee' ,axis = 1,inplace = True)
    test.drop(labels = 'admin_fee' ,axis = 1,inplace = True)
    
    train = pd.merge(train, pd.merge(latlon, address, how='inner', on = 'address'), on = 'ticket_id')
    test = pd.merge(test, pd.merge(latlon, address, how='inner', on = 'address'), on = 'ticket_id')
    
    remove = ['balance_due', 'payment_status', 'compliance_detail', 'agency_name', 'inspector_name', 
              'violator_name', 'violation_street_name', 'mailing_address_str_name', 'city', 'violation_street_number',
              'state', 'zip_code', 'country',  'ticket_issued_date', 'hearing_date', 'violation_description',
              'discount_amount', 'payment_amount', 'disposition', 'address', 'violation_code', 'mailing_address_str_number']

    removetest = ['non_us_str_code', 'agency_name', 'inspector_name', 'violator_name', 'violation_street_name', 
                  'mailing_address_str_name', 'city', 'violation_street_number','state', 'zip_code', 'country', 
                  'ticket_issued_date', 'hearing_date', 'violation_description', 'discount_amount',
                 'grafitti_status', 'violation_zip_code', 'disposition', 'address', 'violation_code', 'mailing_address_str_number']
    
    train.drop(remove, axis = 1, inplace = True)
    test.drop(removetest, axis = 1, inplace = True)
    
    test['lat'] = test['lat'].fillna(test['lat'].mean())
    train['lat'] = train['lat'].fillna(train['lat'].mean())

    test['lon'] = test['lon'].fillna(test['lon'].mean())
    train['lon'] = train['lon'].fillna(train['lon'].mean())
    
    y_train = train['compliance']
    X_train = train.drop(['compliance'], axis=1)
    X_test = test
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    reg = RandomForestRegressor(max_depth = 10,random_state=0).fit(X_train, y_train)
    grid_values = {'n_estimators': [10, 100], 'max_depth': [None, 30]}
    grid_clf_auc = GridSearchCV(reg, param_grid=grid_values, scoring='roc_auc')
    grid_clf_auc.fit(X_train, y_train)
    
    df = pd.DataFrame(grid_clf_auc.predict(test), test.ticket_id)
    return df
#blight_model()


# In[ ]:



