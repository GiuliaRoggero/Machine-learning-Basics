import numpy as np 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
iris = load_iris()

dir(iris)
iris.target_names

#guardo un attimo la distribuzione dei dati 

setosa = df[df["target"]==0]
versicolor = df[df["target"]==1]
virginica = df[df["target"]==2]

#vediamo come si distribuiscono le features


plt.scatter(setosa[0],setosa[1],color="blue")
plt.scatter(versicolor[0],versicolor[1],color="black")

plt.scatter(versicolor[1],versicolor[2],color="red")
plt.scatter(virginica[1],virginica[2],color="green")


l_scores = cross_val_score(LogisticRegression(),iris.data,iris.target)

np.average(l_scores)

d_scores= cross_val_score(DecisionTreeClassifier(),iris.data,iris.target)

np.average(d_scores)

s_scores= cross_val_score(RandomForestClassifier(),iris.data,iris.target)

np.average(s_scores)

r_scores= cross_val_score(SVC(kernel="linear"),iris.data,iris.target)

np.average(r_scores)

#il migliore è il support vector machine con kernel = "Linear"

#guardando come si distribuivano i dati era quasi ovvio che il support vector machine sarebbe stato il migliore














