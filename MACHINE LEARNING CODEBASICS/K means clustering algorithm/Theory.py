#QUESTO è UNSUPERVISED LEARNING! 
#HO SOLO LE FEATURES, NON HO LA COLONNA TARGET 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
#k means is a very popular cluster algorithm
#k is a free parameter

#adjust centroid for the k clusters

#what is a good number of k?  there is a technique for that!

#calcolo per ogni cluster SSE sum of squared errors, SSE = SSE_1 + SSE_2...

#elbow techinque

df = pd.read_csv("income.csv")
df.head()


plt.scatter(df["Age"],df["Income($)"],color="red",marker="*")


#vedo un po come si distribuiscono i dati per trovare i cluster

#scegliere k qui è facile! 

km = KMeans(n_clusters=3)
#ignoro ovviamente la colonna con i nomi. le stringhe non mi interessano

y_predicted = km.fit_predict(df[["Age","Income($)"]]) #non ho la colonna target

#avro tre clusters 0,1,2

df["cluster"] = y_predicted


df0 = df[df.cluster == 0]
df1 = df[df.cluster == 1]
df2 = df[df.cluster == 2]

plt.scatter(df0.Age,df0["Income($)"],color="blue")
plt.scatter(df1.Age,df1["Income($)"],color="green")
plt.scatter(df2.Age,df2["Income($)"],color="red")

plt.xlabel("Age")
plt.ylabel("Income($)")
plt.legend()

#I HAVE TO SCALE MY FEATURES OTHERWISE I GET THIS PROBLEM
#let's do some preprocessing

scaler =MinMaxScaler()
scaler.fit(df[["Income($)"]])
df["Income($)"] = scaler.transform(df[["Income($)"]])

#faccio stessa cosa per age

scaler.fit(df[["Age"]])
df["Age"] = scaler.transform(df[["Age"]])


#train scale dataset

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[["Age","Income($)"]])

df["cluster"] = y_predicted 

df0 = df[df.cluster == 0]
df1 = df[df.cluster == 1]
df2 = df[df.cluster == 2]


km.cluster_centers_ #sono i miei centroidi

plt.scatter(df0.Age,df0["Income($)"],color="blue")
plt.scatter(df1.Age,df1["Income($)"],color="green")
plt.scatter(df2.Age,df2["Income($)"],color="red")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="purple",marker="+",label="centroid")
#now I have a good cluster

#elbow plot technique
#per vedere quale è il migliore K da scegliere

k_rng = range(1,10) #li metto io, k da 1 a 10
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[["Age","Income($)"]])
    sse.append(km.inertia_) #SUM OF SQUARED ERROR
    
plt.xlabel("K")
plt.ylabel("sum of squared errors")
plt.plot(k_rng,sse)

#infatti come noto a k=3 è il nostro Elbow, l'angolo






















