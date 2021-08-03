import numpy as np
import pandas as pd
#titanic example
#why is it called naive? cause we make a naive assumption that features such as male
#class,age,cabin,fare etc are independent of each other 
#in reality some features are related... but doesn't matter now

#naive makes the algorithm really simple

df = pd.read_csv("titanic.csv")
#rimuovo ultima riga inutile
df = df[:-1]
#vediamo i dati che ci mancano per farci un idea.. agee e cabin mancano molti dati
df.isnull().sum()

df.head(10)


df.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis=1,inplace=True)


target = df.Survived
inputs = df.drop("Survived",axis=1)

#we want to convert text, male female, into dummy variables

dummies = pd.get_dummies(inputs.Sex)
dummies.head()


inputs = pd.concat([inputs,dummies],axis=1)
inputs.head()

inputs.drop("Sex",axis=1,inplace=True)
inputs.head()


inputs.columns[inputs.isna().any()] #ci dice dove sono valori nulli

inputs.Age[:10] #Vedo gia che ci sono valori nulli

#prendo la media dei valori e uso quello per coprire i valori nulli
inputs.Age.mean() #uesta è la media

inputs.Age = inputs.Age.fillna(inputs.Age.mean())

inputs.isnull().sum() #non ci sono piu valori nulli


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(inputs,target,test_size=0.2)
len(X_train)
len(X_test)


#now CREATE NAIVE BAYES MODEL

from sklearn.naive_bayes import GaussianNB #when data distribution is normal!
#we can use this model here 
model = GaussianNB()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

model.score(X_test,y_test) #it's not very high

X_test
y_test

#1=survived, 0=not survived

model.predict_proba(X_test[:10])










