import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv("canada_per_capita_income.csv")

plt.scatter(df.year,df["per capita income (US$)"],color="black")
plt.xlabel("year")
plt.ylabel("per capita income (US$)")


#applico retta di regressione lineare


reg = LinearRegression()

reg.fit(df[["year"]],df["per capita income (US$)"])


reg.predict([[2011]])
reg.predict([[2010]])
reg.predict([[2020]])
################################solo per capire come è fatta la retta guardo sotto
reg.predict([[1970]])
reg.predict([[1960]])
reg.predict([[0]])
################################
coefficente = reg.coef_
intercetta = reg.intercept_

#il coefficente della retta di regressione è positivo 


#guardo un po come si mette nel grafico


plt.scatter(df.year,df["per capita income (US$)"],color="black")
plt.plot(df.year,reg.predict(df[["year"]]),color="red")
plt.xlabel("year")
plt.ylabel("per capita income (US$)")

#perfetto :)


income = 828.46507522*2020 - 1632210.7578554575

