import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC

digits = load_digits()

dir(digits)

df = pd.DataFrame(digits.data)

df["target"] = digits.target

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df.drop("target",axis=1),df.target,test_size=0.3,random_state=42)

model_linear = SVC(kernel="linear")

model_linear.fit(X_train,y_train)


y_pred = model_linear.predict(X_test)

model_linear.score(X_test,y_test)

model_rbf=SVC(kernel="rbf") #migliore

y_pred = model_rbf.predict(X_test)

model_rbf.fit(X_train,y_train)

model_rbf.score(X_test,y_test)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)


















