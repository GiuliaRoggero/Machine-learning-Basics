import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("carprices.csv")

df.head()

pd.get_dummies(df["Car Model"])

dummies= pd.get_dummies(df["Car Model"])


merged = pd.concat([df,dummies],axis=1)


final = merged.drop(["Car Model","Audi A5"],axis=1)


lr = LinearRegression()
X=final.drop("Sell Price($)",axis=1)
y=final["Sell Price($)"]


lr.fit(X,y)


lr.coef_
lr.intercept_


lr.score(X,y)

#predizioni


lr.predict([[45000,4,0,1]])

lr.predict([[86000,7,1,0]])



#giusto, inoltre uso get_dummies che è molto piu facile! 
























