#Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

#Creating The datasets
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

initial_features=train.columns.tolist()

del train["PassengerId"]
del test["PassengerId"]
del train["Name"]; del train["Ticket"]
del test["Name"]; del test["Ticket"]
del train["Cabin"]; del test["Cabin"]



#Create a Target Variables
target=train.iloc[:,0:1]
numeric_features=train.select_dtypes(include=[np.number])
categorical_features=train.select_dtypes(exclude=[np.number]).columns

#Taking Care of null values for Categorical Data
numeric_features=train.select_dtypes(include=[np.number])
categorical_features=train.select_dtypes(exclude=[np.number]).columns

train[categorical_features]=train[categorical_features].fillna("Not Availaible")
test[categorical_features]=test[categorical_features].fillna("Not Availaible")

del train["Survived"]

#Refreshing the Numerical and categorical values
numeric_features=train.select_dtypes(include=[np.number])
categorical_features=train.select_dtypes(exclude=[np.number]).columns


#Encoding Categorical Data:
for i in categorical_features:
    l_train=train[i].unique().tolist()
    l_test=test[i].unique().tolist()
    l_train.sort()
    l_test.sort()
    if(l_train==l_test):
        train=pd.get_dummies(train,columns=[i],drop_first=True)
        test=pd.get_dummies(test,columns=[i],drop_first=True)
    else:
        l1=len(train.columns)
        l2=len(test.columns)
        dummy_train=pd.get_dummies(train,columns=[i])
        dummy_test=pd.get_dummies(test,columns=[i])        
        dummy_train=dummy_train.iloc[:,l1-1:] 
        dummy_test=dummy_test.iloc[:,l2-1:]
        col_train_dummy=dummy_train.columns.tolist()
        col_test_dummy=dummy_test.columns.tolist()
        itr=0
        for j in col_train_dummy:
            if(j not in col_test_dummy):
                dummy_test.insert(itr,j,0)
                col_test_dummy.append(j)
            itr+=1
        itr=0
        for j in col_test_dummy:
            if(j not in col_train_dummy):
                dummy_train.insert(itr,j,0)
                col_train_dummy.append(j)
            itr+=1
        train=train.join(dummy_train)
        test=test.join(dummy_test)
        del train[i]
        del test[i]

l_train=train.columns.tolist()
l_test=test.columns.tolist()

del train["Embarked_Not Availaible"]; del test["Embarked_Not Availaible"]

#Creating The Numpy Arrays  
X_train=train.iloc[:,:].values
y_train=target
X_test=test.iloc[:,:].values

#Handling the Null values of Numeric features
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_train=imputer.fit_transform(X_train)
X_test=imputer.fit_transform(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

pd.DataFrame(y_pred, columns=['Survived']).to_csv('Titanicpred_final2.csv')
