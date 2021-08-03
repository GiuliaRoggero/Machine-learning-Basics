import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from sklearn.datasets import load_digits

digits = load_digits()

dir(digits)


digits.data[0] #la prima riga 

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
    
    

digits.target[0:5]


#we use data and target to train our model 


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.2,random_state=42)
len(X_train)
len(X_test)
model = LogisticRegression()

classifier = model.fit(X_train,y_train)


model.score(X_test,y_test) #è molto alto

#faccio predizioni adesso 

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test,y_pred) #funziona molto bene 



#faccio degli esempi 

plt.matshow(digits.images[67]) #questo qua sembra che mi dovrebbe dare 6 nella predizione

digits.target[67] #mi da il numero 6

#vediamo ora se il modello me lo predice in modo corretto


model.predict([digits.data[67]]) #giusto predice 6 che è giusto

model.predict(digits.data[0:5]) #predice giusto

#digits.data[0:5] non ha bisogno di altre parentesi quadre dato che scritto cosi è gia una dataframe
#questo è il modo veloce di plottare la confusion matrix
plot_confusion_matrix(classifier,X_test,y_test) #in questo modo la metto come grafico

#oppure posso fare cosi 
#sns.heatmap!!! 

import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True) #annot per mettere valori numerici all'interno dei quadrati 
plt.xlabel("predicted")
plt.ylabel("true")




































