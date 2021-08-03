#overfitting is a very common problem in machine learning
#L1 and L1 regularization can be used for overfitting issues

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#we will see how the accuracy score of a model can improve using L1 and L2 regularization
#to reduce the problem of "overfitting"

dataset = pd.read_csv("Melbourne_housing_FULL.csv")
dataset.head()

dataset.nunique() #nunique per vedere tutti i valori unici 

dataset.shape #per vedere che tipo di matrice è

#posso eliminare le colonne che non servono

#queste sono le colonne utili per l'analisi:

col_to_use = ["Suburb","Rooms","Type","Method","SellerG","Regionname","Propertycount",
              "Distance","CouncilArea","Bedroom2","Bathroom","Car","Landsize","BuildingArea",
              "Price"]

dataset = dataset[col_to_use]
dataset.head()

#now I have 15 columns instead of 21

dataset.isnull().sum()

dataset = dataset[:-1] #remove last row
#there are a lot of missing data!
#we need to handle this 

cols_to_fill_zero = ["Propertycount","Distance","Bedroom2","Bathroom","Car"]
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)

dataset.isnull().sum()

#ok now the other ones
# [landsize and building area] fill with the mean

dataset["Landsize"] = dataset["Landsize"].fillna(dataset.Landsize.mean())
dataset["BuildingArea"] = dataset["BuildingArea"].fillna(dataset.BuildingArea.mean())

dataset.isnull().sum()


#now all the independent variable are filled , for the dependent i can drop some rows
#the dataset is huge, it's not a big deal

#i can drop the row with the nan values in regionname and councilarea
#the dataset is huge, nothing bad will happen

dataset.dropna(inplace=True)

dataset.isnull().sum()

from sklearn.linear_model import LinearRegression

#now dummy variables
#for every text columns
#one hot encoding
dataset = pd.get_dummies(dataset,drop_first=True) #drop first to drop one dummy columns to prevent the dummy trap
#i want to predict the price
X = dataset.drop("Price",axis=1)
y= dataset["Price"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#now let's do regular linear regression
reg = LinearRegression()

reg.fit(X_train,y_train)


y_pred = reg.predict(X_test)


reg.score(X_test,y_test) #it's really low

reg.score(X_train,y_train) #this is higher..why? BECAUSE THE MODEL IS OVERFITTED

#let's use the ridge and lasso regression
#lasso = L1
#rigde = L2
from sklearn.linear_model import Lasso



lasso_reg = Lasso(alpha = 50,max_iter = 1000,tol = 0.1) #alpha = 0, same as linear regression

lasso_reg.fit(X_train,y_train)

lasso_reg.score(X_test,y_test) #score is a little higher 


from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha = 50,max_iter=1000,tol=0.1)

ridge_reg.fit(X_train,y_train)

ridge_reg.score(X_test,y_test)







