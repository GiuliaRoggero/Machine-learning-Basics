import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris

iris = load_iris()
#questi sono i tipi di 3 fiori che posso avere
#0 setosa
#1 versicolor
#2 virginica 


df = pd.DataFrame(iris.data,columns= iris.feature_names)
print(df.head())

df["target"] = iris.target #aggiungo una colonna al dataset

df["target"].replace({0:"Setosa",1:"versicolor",2:"virginica"},inplace=True)

X= df.drop(["target"],axis=1)
y= df["target"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.4,random_state=42)

len(X_train)
len(X_test)

model = LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)


model.score(X_test,y_test)


model.predict([[6.3,2.7,4.9,1.8]]) #virginica! 

cm=confusion_matrix(y_test,y_pred)
#confusion matrix

import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel("predicted")
plt.ylabel("true")


#perfetto!!!!! 


























