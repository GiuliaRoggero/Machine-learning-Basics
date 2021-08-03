import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#CLASSIFICATION PROBLEMS
#WE USE LOGISTIC REGRESSION FOR THIS
#BINARY CLASSIFICATION, YES OR NOT
#MULTICLASS CLASSFICATION IF THERE ARE MORE CLASSES, THAT NEED TO BE CLASSIFIED

df = pd.read_csv("insurance_data.csv")
df.head(10)
#vediamo come si distribuiscono i dati

plt.scatter(df.age,df["bought_insurance"],color="blue")

#come vedo segue il modello classico della logistic regression
#non è un retta come nella linear regression, ma è una sigmoid or logit function
#con la forma ad S sdraiata

#sigmoid(z) = (1)/(1 + e^-z) e = 2,71
#sigmoid function converts input into range 0 to 1

#essentially we are doing this
# y = m*x + b che è la linear regression
# fitto questa equazione in una sigmoid function  y = (1)/(1 + e^-(m*x + b))
#è questa è l'equazione della linea della logistic regression (s-shape line)

#split the dataset

X=df[["age"]]
y= df["bought_insurance"]


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.9,random_state=42)


len(X_test)
len(X_train)


model = LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

model.score(X_test,y_test) #in questo caso ho 1, gli ho azzeccati tutti
#ma succede perchè il dataset è molto piccolo
#QUINDI ABBIAMO UNO SCORE ALTO SOLO PERCHè ABBIAMO POCHI DATI

cm = confusion_matrix(y_test,y_pred) #come vedo sono tutti azzeccati

model.predict([[62]]) #si ha l'assicurazione
model.predict([[1]]) # no, non ha l'assicurazione
#e cosi via...


model.predict_proba([[1]]) #molta alta la probabilitàò che non abbia l'assicurazione
model.predict_proba([[62]]) #molto probabile che abbia l'assicurazione


  








































