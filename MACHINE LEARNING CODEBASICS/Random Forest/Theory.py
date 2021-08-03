import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

dir(digits)

plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])  #matshow rende una matrice una figura! plt.gray() per averlo tutto grigio
    

df = pd.DataFrame(digits.data)
df.head()


target = pd.Series(digits.target)


df["target"] = digits.target

X_train,X_test,y_train,y_test = train_test_split(df.drop("target",axis=1),digits.target,test_size=0.2,random_state=42)

len(X_train)
len(X_test)

from sklearn.ensemble import RandomForestClassifier #emseble is when we are using multiple algorithms
#multiple trees to find final output

model = RandomForestClassifier(n_estimators=60)

model.fit(X_train,y_train)

#n_estimators = 10 cioe usa 10 decision trees

model.score(X_test,y_test)

y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
 
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("True")












