import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv("titanic.csv")

df.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis=1,inplace=True)

#survive è la nostra colonna che vogliamo sapere 

inputs = df.drop(["Survived"],axis=1)
target = df.Survived

inputs.isnull().sum() #nel target abbiamo molti valori nulli soprattuto in age che dobbiamo risolvere
target.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le_Gender = LabelEncoder()

inputs = inputs[:-1] #per eliminar ultima riga che era solo con NAN
target = target[:-1]

inputs["Gender"] = le_Gender.fit_transform(inputs["Sex"])

inputs  = inputs.drop("Sex",axis=1)

#ora tolgo i valori nulli della colonna age 

inputs.isnull().sum()
inputs.Age = inputs.Age.fillna(inputs.Age.mean())

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(inputs,target,test_size=0.2,random_state=42)

len(X_train)
len(X_test)

model = DecisionTreeClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)


model.score(X_test,y_test)


from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test,y_pred)















