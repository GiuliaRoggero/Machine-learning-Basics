import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from numpy import array
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LinearRegression
# F9 per runnare solo una riga

#come troviamo la linea di regressione?
#semplicemente  minimaze (sum (da 1 a n) delta(i)^2) che sono le distanze dei punti (veri) dalla retta
#di regressione
#faccio questa cosa per tutte le linee di regressione, e trovero quella con L'errore minimo
#questa sarà la nostra retta di regressione

df= pd.read_csv("homeprices.csv")

#nel nostro caso: area è l'independent variable
#price è la variabile dipendente

# price = m*area + b

#plot a scatter plot to see the distribution of the data

plt.scatter(df.area,df.price,color="black",marker="+")
plt.xlabel("area(srt ft)")
plt.ylabel("price(US$)")
#vedo che i dati sembra che seguano una linea, con coefficente positivo
#quindi una regressione lineare sembra più che giusta in questo caso

###########################################################
#nota bene
# c = df.price è una Series
# c = df[["price"]] è un DataFrame
###########################################################

reg = LinearRegression() #creo il nostro object

reg.fit(df[["area"]],df.price) #a sinistra devo mettere un DataFrame, a destra una Series
#ora il nostro modello è allenato

#adesso posso predirre cosa voglio 

reg.predict([[3600]]) #devo metterci un dataframe dentro questo argomento, per questo metto [[]]


#let's look at the iternal details

print(reg.coef_)
print(reg.intercept_)

#questi sono i dati della retta di regressione che abbiamo costruito

#     y = m*x + b     m=135.78767123 and b= 180616.43835616432
#     price =  135.78767123*area +  180616.43835616432

#infatti controprova


price =  3600*135.78767123 + 180616.43835616432 #che è lo stesso numero di reg.predict([[3600]])

reg.predict([[5000],[3300],[3600]]) #cosi posso fare piu predizione contemporaneamente

#GRAFICO DELLA RETTA DI REGRESSIONE

plt.scatter(df.area,df.price,color="black",marker="+")
plt.plot(df.area,reg.predict(df[["area"]]),color="red")
plt.xlabel("area",fontsize=20)
plt.ylabel("price",fontsize=20)


#adesso che ho allenato il modello posso fare predizioni su tutto, ovviamente
#dando per scontato che anche i nuovi dati si muovino come con quelli con cui ho allenato 
#il modello

#predico i prezzi di queste case per esempio date le relative aree

areas = pd.read_csv("areas.csv")
areas.head(5)

reg.predict(areas) #predict legge i dataframe, quindi posso direttamente infilare 
#il dataframe areas

prices = reg.predict(areas) #in questo modo è un array

#lo trasformo in una Series

prices = Series(prices)

areas["prices"] = prices #facendo cosi creo una nuova colonna nel dataframe

#esportare i dati

#rinomino il file

predictions = areas
#esportare i file, molto importante
predictions.to_csv("predictions.csv")
predictions.to_excel("predictions.xlsx",index=False) #index=False per togliere gli indici di fianco



















































