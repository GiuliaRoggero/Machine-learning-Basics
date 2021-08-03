import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv("HR_comma_sep.csv")
df.head()


#DATA EXPLORATION AND VISUALISATION

left = df[df.left==1]

retained = df[df.left==0]

left.shape 
retained.shape #sono di piu che sono restate di quelle che sono andate via


num_cols = ['satisfaction_level', 'last_evaluation']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

group = df.groupby("left").mean()

#guardo quali sono le variabili che cambiano di piu tra chi resta e chi se ne va via
#1 satisfaction_level cambia molto in chi resta e chi va
#2 average_monthly_hours stessa cosa
#3 promotion last 5 years anche

#guardiamo adesso anche le variabili non numeriche: salary e deparment


pd.crosstab(df.salary,df.left).plot(kind="bar")

#people with high salary usually don't leave the job

pd.crosstab(df.Department,df.left).plot(kind="bar")
#questo possiamo ignorarlo

#quindi abbaimo 4 variabili importanti
#1 satisfaction level
#2 average monthly hours
#3 promotion last 5 years
#4 salary

subdf  = df[["satisfaction_level","average_montly_hours","promotion_last_5years","salary"]]
subdf.head()

salary_dummies = pd.get_dummies(df["salary"],prefix="salary")

df_with_dummies = pd.concat([subdf,salary_dummies],axis=1)

df_with_dummies = df_with_dummies.drop("salary",axis=1)

df_with_dummies= df_with_dummies.drop("salary_high",axis=1)

X= df_with_dummies
y= df.left
#droppo le linee con valori nulli

X= X.dropna()
y= y.dropna()



X.isnull().sum()
y.isnull().sum()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.7,random_state=42)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

esempio = model.predict([[0.09,254,0,1,0]]) 

#score e accuracy_score sono la stessa cosa infatti
##############################################################
model.score(X_test,y_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
#############################################################à

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)


#nota: anche in questo caso io toglierei una dummy variable per non ricadere nella dummy trap

