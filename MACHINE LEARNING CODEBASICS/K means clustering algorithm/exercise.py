from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()

df["flower"] = iris.target
df.head()

df.drop(["sepal length (cm)","sepal width (cm)","flower"],axis=1,inplace=True)

#mi rimangono solo due features

km = KMeans(n_clusters=3) #infatti se ci penso ho solo tre tipi di fiore che posso avere come target


yp= km.fit_predict(df)

df["cluster"] = yp

df.cluster.unique() #avro tre cluster

df0 = df[df.cluster ==0]
df1 = df[df.cluster ==1]
df2 = df[df.cluster ==2]

plt.scatter(df0["petal length (cm)"],df0["petal width (cm)"],color="blue")
plt.scatter(df1["petal length (cm)"],df1["petal width (cm)"],color="black")
plt.scatter(df2["petal length (cm)"],df2["petal width (cm)"],color="red")

#come vedo 3 clusters sono giusti, ora controllo per sicurezza il numero perfetto di clusters
# con l'elbow technique

sse = []
k_rng = range (1,10)
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(df)
    sse.append(km.inertia_) #sum of squared errors
    
    
plt.xlabel("K")
plt.ylabel("sum of squared errors")
plt.plot(k_rng,sse)


#K = 3 è il numero migliore






