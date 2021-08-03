import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#allenare il modello con un intero dataset non è una buona strategia
#è meglio splittare il dataset in due parti 
#il training data set e il test data set
#il training deve essere piu grande di solito (tipo 80,70 percento)
#il test è il rimanente

df = pd.read_csv("carprices.csv")
df.head(10)

#vediamo un po come si distribuiscono i dati , le due variabili indipendenti con la variabile dipendente

plt.scatter(df.Mileage,df["Sell Price($)"],marker=".",color="black")
#si distribuisce in modo lineare, piu aumenta il mileage piu diminuisce il prezzo

plt.scatter(df["Age(yrs)"],df["Sell Price($)"],marker=".",color="black")
#anche questi dati sono distribuiti in modo lineare con coefficente angolare negativo

#POSSO ALLORA APPLICARE REGRESSIONE LINEARE MULTIVARIATA (DATO CHE CI SONO DUE VARIABILI)

X = df[["Mileage","Age(yrs)"]]
y= df["Sell Price($)"]

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42) #quindi il training data è 80 percento
#random state è per non far cambiare X_train.. ogni volta che compilo
len(X_train)
len(X_test)


lr = LinearRegression()
lr.fit(X_train,y_train)

lr.score(X_train,y_train)

y_pred = lr.predict(X_test)

lr.score(X_test,y_test)

#from sklearn.metrics import accuracy_score
# accuracy_score(y_test,y_pred) #non posso usare accuracy_score per la regressione ma solo per classificazione

#NB CONFUSION MATRICES ARE NOT FOR REGRESSION, ONLY FOR CLASSIFICATION


















