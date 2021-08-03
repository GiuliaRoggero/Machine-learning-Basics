import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#predict home prices from area, bedrooms and age (multiple variables)

df = pd.read_csv("homeprices.csv")

#there is a missing data point!
#infatti:
    
df.isnull()

df.isnull().sum()

#c'è relazione lineare tra prezzo e area, prezzo e bedrooms e prezzo e age
#quindi posso usare la linear regression multiple variables

plt.scatter(df.area,df.price) #cresce l'area, cresce il prezzo in modo piu o meno lineare

plt.scatter(df.bedrooms,df.price) #cresce numero di bedrooms, cresce prezzo in modo piu o meno lineare

plt.scatter(df.age,df.price) #cresce l'eta della casa, diminuisce il prezzo, in modo piu o meno lineare

#allora posso usare la regressione lineare
#avrà questo aspetto price = m_1*area + m_2*bedrooms + m_3*age + b 
#x_1, x_2, x_3 sono le features, le m sono i coefficenti e b è l'intercetta

#prima di tutto devo togliere quel valore nullo, come faccio?

#il valore nullo si trova nella colonna bedrooms.. posso 
#quindi prendere la mediana dei valori e piazzare il valore trovato 
#al posto del valore nullo

median_bedrooms =df.bedrooms.median() #la mediana è 4

df.bedrooms = df.bedrooms.fillna(median_bedrooms)

#ho quindi sostituito il valore nullo con la mediana della colonna


#ora non ho piu valori nulli 

reg = LinearRegression()

reg.fit(df[["area","bedrooms","age"]],df.price) #now the model is ready

reg.coef_
reg.intercept_


#quindi è come scrivere

# price = 112.06244194*area + 23388.88007794*bedrooms -3231.71790863*age + 221323.00186540408

reg.predict([[3000,3,40]]) #ato che la casa ha 40 anni costerà non tanto
#stesso risultato con 
risultatoprezzo = 112.06244194*3000 + 23388.88007794*3 + -3231.71790863*40 + 221323.00186540408

reg.predict([[2500,4,5]]) # il prezzo è abbastanza alto perchè la casa è praticamente nuova
























