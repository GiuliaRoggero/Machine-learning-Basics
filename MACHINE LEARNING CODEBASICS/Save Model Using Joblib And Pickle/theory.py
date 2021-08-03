#often the size of the dataset is pretty huge
#as the size of the dataset increases the model become more accurate

#how to save a train model to a file?
#in pratica voglio salvare il modello allenato

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("homeprices.csv")
df.head()


model = LinearRegression()
model.fit(df[["area"]],df.price)
model.coef_
model.intercept_
model.predict([[5000]])

#ho costruito il mio modello
#come lo salvo il mio modello allenato?
#COSI

import pickle
with open("model_pickle","wb") as f:
    pickle.dump(model,f)
#now model is a saved in my working directory
with open("model_pickle","rb") as f:
      mp  = pickle.load(f)
      
mp.predict([[5000]]) # è la stessa cosa del modello di prima



#second approach to save a model  sklearn joblib

import joblib
joblib.dump(model,"filename.pkl")


modelloallenato = joblib.load("filename.pkl")
modelloallenato.predict([[5000]])

#joblib è piu facile piu corto, ed è piu efficente per trasportare un numero largo di array




















