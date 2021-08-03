import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#2 years experience,9 test score, 6 interview score -> predict salary
#12 years experience, 10 test score, 10 interview score -> predict salary

df = pd.read_csv("hiring.csv")

#prima di tutto riempio i Nan che non ci permettono di studiare i dati 
#nel machine learning

df.isnull().sum()
#il proff ha detto che posso assumerli a zero quelli dell'esperienza
df.experience.fillna("zero")

df.experience = df.experience.fillna("zero")

#per l'altro valore nullo utilizzo la media (del test_score)


import math

median_test_Score = math.floor(df["test_score(out of 10)"].mean())
df["test_score(out of 10)"] = df["test_score(out of 10)"].fillna(median_test_Score)

#trasformo la parola in numero!

from word2number import w2n

df.experience = df.experience.apply(w2n.word_to_num)


#ora posso studiare i dati

plt.scatter(df.experience,df["salary($)"]) #cresce esperienza cresce salario in modo lineare
plt.scatter(df["test_score(out of 10)"],df["salary($)"])
plt.scatter(df["interview_score(out of 10)"],df["salary($)"])


reg = LinearRegression()

reg.fit(df[["experience","test_score(out of 10)","interview_score(out of 10)"]],df["salary($)"])

reg.coef_ #sono tutti coefficenti angolari positivi, cresce una features e cresce il salario
reg.intercept_


reg.predict([[2,9,6]])

#il salario deve essere molto grande per quello sotto:
reg.predict([[12,10,10]])























