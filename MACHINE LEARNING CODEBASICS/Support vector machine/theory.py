#support vector machine 
#very popular classification algorithm


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


iris = load_iris()

dir(iris)
iris.feature_names

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df["target"] = iris.target #aggiungo la colonna al dataframe df con la colonna target

iris.target_names #0 =setosa, 1 = versicolor, 2 = virginica

df[df.target ==1].head() #ci sono 50 versicolor
df[df.target ==0].head() #ci sono 50 setosa
df[df.target ==2].head() #ci sono 50 virginica


df["flower_name"] = df.target.apply(lambda x: iris.target_names[x])
 

#DATA VISUALIZATION; IMPORTANT

df0 = df[df.target ==0] #setosa
df1 = df[df.target ==1] #versicolor
df2 = df[df.target ==2] #virginica

#draw scatterplot
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.scatter(df0["sepal length (cm)"],df0["sepal width (cm)"],color="green",marker="+")
plt.scatter(df1["sepal length (cm)"],df1["sepal width (cm)"],color="blue",marker=".")
#there is a clear evidence

plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.scatter(df0["petal length (cm)"],df0["petal width (cm)"],color="green",marker="+")
plt.scatter(df1["petal length (cm)"],df1["petal width (cm)"],color="blue",marker=".")
#even clear boundery

#the support vector machine will work really well 

#nella realta lo farei per tutte le features! 

from sklearn.model_selection import train_test_split

X = df.drop(["target","flower_name"],axis=1)
y = df.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


len(X_train)
len(X_test)


from sklearn.svm import SVC

model = SVC() #se metto C=10000 il mio score diminuisce
              #posso anche usare kernel = "Linear"


model.fit(X_train,y_train) #C is the regularization parameter, gamma , kernel..

y_pred = model.predict(X_test)

model.score(X_test,y_test) #molto alto! 










