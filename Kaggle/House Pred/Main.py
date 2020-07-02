#Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Creating The datasets
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#Dropping missing values with threshold 70%
train.dropna(thresh=1000,axis=1,inplace=True)
new_cols=train.columns
old_cols=test.columns
print("\nColumns Having Threshold NAN > 70%\n")
for i in old_cols:
    if(i not in new_cols):
        print(i,"\n")
        del test[i]
print("\n***************************************************************\n")

#Create a Target Variable
target=train.iloc[:,-1:].values
numeric_features=train.select_dtypes(include=[np.number])
numeric_features.dtypes
categorical_features=train.select_dtypes(exclude=[np.number])

#Finding features with Best correaltion to target data
corr=numeric_features.corr()
print()
print(corr["SalePrice"].sort_values(ascending=False))
print("\nCorelation of the different Features with the Final output is:\n")
print("\n***************************************************************\n")
#Remove highly correlated features and Low corelated features
col_corr=set()
for i in range(len(corr.columns)):
    for j in range(i):
        #Removes Highly Corelated Features (Eachother)
        if (corr.iloc[i,j] >=0.8) and (corr.columns[j] not in col_corr):
            colname = corr.columns[i] #getting the name of column
            col_corr.add(colname)
            if colname in train.columns:
                del train[colname]
                del test[colname]
        #Removes Less corelated Features (SalePrice)
        if(i==len(corr.columns)-1 and corr.iloc[i,j]<=0.3):
            colname = corr.columns[j] #getting the name of column
            col_corr.add(colname)
            if colname in train.columns:
                del train[colname]
                del test[colname]

print("The deleted Columns are:\n")
for i in col_corr:
    print(i,"\n")
print("\n***************************************************************\n")
#Refreshing the Numerical and categorical values
numeric_features=train.select_dtypes(include=[np.number])
categorical_features=train.select_dtypes(exclude=[np.number]).columns

#Taking Care of null values for Categorical Data
train[categorical_features]=train[categorical_features].fillna("Not Availaible")
test[categorical_features]=test[categorical_features].fillna("Not Availaible")
    
del train["SalePrice"]

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
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_train=sc.fit_transform(y_train)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)
y_pred=sc.inverse_transform(y_pred)

pd.DataFrame(y_pred, columns=['SalePrice']).to_csv('Housing_pred.csv')
