import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
#naive bayes classifier is perfect for email spam detection

wine = load_wine()

dir(wine)


df= pd.DataFrame(wine.data,columns= wine.feature_names)
target= pd.DataFrame(wine.target)

df["target"] = wine.target

X_train,X_test,y_train,y_test = train_test_split(wine.data,wine.target,test_size=0.3,random_state=42)


from sklearn.naive_bayes import GaussianNB,MultinomialNB

model1 = GaussianNB()

model1.fit(X_train,y_train)

y_pred = model1.predict(X_test)

model1.score(X_test,y_test) #altissimo!

model2 = MultinomialNB()
model2.fit(X_train,y_train)

y_pred1= model2.predict(X_test)

model2.score(X_test,y_test) #meno alto di prima

#giusto perchè le features sono numeri continui...


