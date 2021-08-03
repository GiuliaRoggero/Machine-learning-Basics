import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()

dir(iris)

df = pd.DataFrame(iris.data)
target = pd.Series(iris.target)
df["target"] = target


from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(df.drop("target",axis=1),target,test_size=0.2)

df.head()

model = RandomForestClassifier(n_estimators=10)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

model.score(X_test,y_test)

model = RandomForestClassifier(n_estimators=40)
model.fit(X_train, y_train)
model.score(X_test,y_test)
