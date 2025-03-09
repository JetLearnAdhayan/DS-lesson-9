#Steps to be followed in a machine learning project 
#a) gathering the data and load it 
#b) preprocess the data - for future machine learning projects
#c) Analyse the data yourself and define your input and output coloumns and remove the rest
#d) define input and output - create separate dataframes for input and output
#e) split 80% of the data for training purposes and 20% for testing
#f) select the machine learning algorithim you're going to use
#g) compare the pridictions with the acutal results to fid out accuracy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Data.csv")

print(dataset.head())
print(dataset.info())

#identifying input(features) and output(target)
#output = purchased or not
#input = country, age, salary

X = dataset.iloc[:, :-1].values  #picked all but last column
y = dataset.iloc[:, -1].values #picked only last coloumn

print("Features :\n",X)
print("Target :\n",y)
print()

#adds missing values to the data by using mean mode or median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
#transform the data for the entire dataset (age and salary alone)
X[:,1:3] = imputer.fit_transform(X[:,1:3])

print("After Imputing :\n",X)

#coverts letters into numbers (objective data into numerical data)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(),[0])],remainder="passthrough")
X = pd.DataFrame(ct.fit_transform(X))
print("One Hot Encoding :\n", X)

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
#for yes is 1 and for no is 0

y = le.fit_transform(y)
print("Label Encoder :\n", y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state=1)

print("Xtrain: \n",X_train)
print("X_test: \n", X_test)
print("Y_train: \n", y_train)
print("Y_test: \n", y_test)

#feature scaling from -1 to +1
#if the difference between the values is huge the system training and testing gets affected
from sklearn.preprocessing  import StandardScaler
sc = StandardScaler()

X_train.iloc[:,1:3] = sc.fit_transform(X_train.iloc[:,1:3])
X_test.iloc[:,1:3] = sc.fit_transform(X_test.iloc[:,1:3])

print("After Scaling the values from -1 to +1: \n")
print(X_train)
print(X_test)



