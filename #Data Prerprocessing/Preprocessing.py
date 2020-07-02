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

#Create a Target Variables
target=train.iloc[:,-1:]
del train['SalePrice']
numeric_features=train.select_dtypes(include=[np.number])
categorical_features=train.select_dtypes(exclude=[np.number]).columns

#Taking Care of null values for Categorical Data
train[categorical_features]=train[categorical_features].fillna("NA")
test[categorical_features]=test[categorical_features].fillna("NA")

#Taking Care of null Values for Numerical Data
for i in numeric_features.columns:
    mean=train[i].mean()
    train[i]=train[i].fillna(mean)
    mean=test[i].mean()
    test[i]=test[i].fillna(mean)

#Remove highly correlated features
col_corr=set()
corr=numeric_features.corr()
for i in range(len(corr.columns)):
    for j in range(i):
        #Removes Highly Corelated Features (Eachother)
        if (corr.iloc[i,j] >=0.85) and (corr.columns[j] not in col_corr):
            colname = corr.columns[i] #getting the name of column
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

train.sort_index(axis=1,ascending=True,inplace=True)
test.sort_index(axis=1,ascending=True,inplace=True)

l_train=train.columns.tolist()
l_test=test.columns.tolist()

if(l_train==l_test):
    print("\nEncoding Sucessful\n")
else:
    print("|nEncoding Error\n")

print("\n***************************************************************\n")

#Removing Not NA Columns
for i in train.columns:
    if "_NA" in i:
        del train[i]
        del test[i]

#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
fs = SelectKBest(score_func=chi2, k=150)
fs.fit(train, target)
train=fs.transform(train)
test=fs.transform(test)

#Creating The Numpy Arrays  
X_train=train
y_train=target
X_test=test

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train=sc_y.fit_transform(y_train)

#Model Training 

pd.DataFrame(y_pred, columns=['SalePrice']).to_csv('Housing_pred3.csv')
