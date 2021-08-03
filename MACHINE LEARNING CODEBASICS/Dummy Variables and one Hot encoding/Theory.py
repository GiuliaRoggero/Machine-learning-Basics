#using pandas with get_dummies
#or
#using sklearn with OneHotEncod
#ricordo che quando creo dummy variables, devo droppare una colonna di una dummy variable
#altrimento ricado nella trappola delle dummy variables
#Dummy Variable Trap:
#The Dummy variable trap is a scenario where there are attributes
# which are highly correlated (Multicollinear) and one variable predicts 
# the value of others. When we use one hot encoding for handling the categorical 
# data, then one dummy variable (attribute) can be predicted with the help of other 
# dummy variables. Hence, one dummy variable is highly correlated with other dummy 
# variables. Using all dummy variables for regression models lead to dummy variable 
# trap. So, the regression models should be designed excluding one dummy variable.
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#how can we deal with text data in numeric model 

df = pd.read_csv("homeprices.csv")

#CATEGORICAL VARIABLES ARE NOMINAL:IF NO ORDER LIKE MALE FEMALE, OR GREEN RED BLUE
#OR ORDINAL IF HAVE SOME SORT OF NUMERICAL ORDER: GRADUATE MASTER PHD (THEY HAVE ORDERS)

dummies = pd.get_dummies(df.town)

#pd.concat to join 2 dataframes
merged = pd.concat([df,dummies],axis=1)


#REGOLA IMPORTANTISSIMA, DEVO SEMPRE DROPPARE UNA COLONNA A CASO DI UNA DUMMY VARIABLE
#ALTRIMENTO ENTRO NELLA TRAPPOLA DELLE DUMMY VARIABLE
#ES 5 COLONNE DUMMY VARIABLE, NE DROPPO UNA E RIMANGO CON 4 COLONNE DUMMY VARIABLE

final= merged.drop(["town","west windsor"],axis=1)

model= LinearRegression()

X = final.drop("price",axis=1)

model.fit(X,final.price)
model.coef_
model.intercept_
model.predict([[5000,1,0],[6000,0,0]])

#sarabbe come scrivere

price= 126.89744141*5000 + -40013.97548914*1 + 0* -14327.56396474 + 249790.36766292533

price=  126.89744141*6000 + 0 + 0 + 249790.36766292533


#let's ee how accurate my model is 

model.score(X,final.price) #compare predicted value with actual values

#another way or finding the dummy variables

df.head(9)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dfle = df

dfle.town = le.fit_transform(dfle.town)



X = df[["town","area"]].values
y = dfle.price
#one hot encoding it's too hard

#ONE HOT ENCODER é PPIU DIFFICILE MA é STESSA COSA
from sklearn.preprocessing import OneHotEncoder













