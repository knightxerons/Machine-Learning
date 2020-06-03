#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#importing the dataset 
data1=pd.read_csv("train.csv")
X=data1.iloc[:,2:12].values
y=data1.iloc[:,1].values

data2=pd.read_csv("test.csv")
X_test=data2.iloc[:,1:11]

#taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X=imputer.fit_transform(X)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X_test=imputer.fit_transform(X_test)

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[2,9])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X=np.delete(X,[0,2,6,10,12],1)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[2,9])], remainder='passthrough')
X_test = np.array(ct.fit_transform(X_test))
X_test=np.delete(X_test,[0,2,6,10,12],1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

X_test= sc.fit_transform(X_test)


#Model Svm
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(X, y)

#Model ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X,y, batch_size = 32, epochs = 100)

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

pd.DataFrame(y_pred, columns=['predictions']).to_csv('ANN_prediction.csv')

#PassengerId Survived 892 1039
