import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#how to choose the best model???? hypertuning = choosing the optimal parameters

iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df["flower"] = iris.target
df["flower"] = df["flower"].apply(lambda x: iris.target_names[x])

X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.3,random_state=42)

#first try svm model 

#FIRST: HOW TO DO THE HYPERTUNING
#SECOND: HOW TO CHOOSE THE BEST MODEL

from sklearn.svm import SVC

model = SVC(kernel="rbf",C=30,gamma="auto")
model.fit(X_train,y_train)
model.score(X_test,y_test) #lo score aumenta

#uso cross validation! cosi almeno non cambia lo score ogni volta che cambio i train e test sets

from sklearn.model_selection import cross_val_score
#1
np.average(cross_val_score(SVC(kernel="linear",C=10,gamma="auto"),iris.data,iris.target,cv=5))

#2
np.average(cross_val_score(SVC(kernel="rbf",C=10,gamma="auto"),iris.data,iris.target,cv=5))

#3
np.average(cross_val_score(SVC(kernel="rbf",C=20,gamma="auto"),iris.data,iris.target,cv=5))

#il secondo è il migliore!!!!


#posso fare un for loop per rendere il tutto piu veloce


kernels = ["rbf","linear"]
C= [1,10,20]
avg_scores= {}
for kval in kernels:
    for cval in C:
        cv_scores = cross_val_score(SVC(kernel=kval,C=cval,gamma="auto"),iris.data,iris.target,cv=5)
        avg_scores[kval + "_" + str(cval)] = np.average(cv_scores)
        
avg_scores

#trovo i best scores! 
#this approach has some issues! if i have 4 parameters there will be too many for loops

#FORTUNATELY SKLEARN PROVIDE GRIDSEARCHCV THAD DO THE SAME THING AS BEFORE BUT FASTER



from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(SVC(gamma="auto"), 
                   {  "C": [1,10,20],
                       "kernel": ["rbf","linear"]}, cv=5,return_train_score=False)

clf.fit(iris.data,iris.target)
clf.cv_results_


df = pd.DataFrame(clf.cv_results_)

dir(clf)
#GRIDSEARCHCV IS REALLY IMPORTANT!!!!

clf.best_estimator_ #vedo qual'è il migliore

clf.best_score_ #vedo anche il suo relativo miglior punteggio

clf.best_params_

#one issue is the computation cost
#adesso funziona perchè il nostro dataset è piccolo


#then I can use RandomizedsearchCV, which has less computational cost

from sklearn.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(SVC(gamma="auto"),
                        { "C": [1,10,20],
                          "kernel": ["rbf","linear"]
                          }, cv=5, return_train_score=False, n_iter=2)
#it will try only two combinations of the parameters

rs.fit(iris.data,iris.target)
pd.DataFrame(rs.cv_results_)[["param_C","param_kernel","mean_test_score"]]
#vedo infatti che prova a caso due combinazione


#HOW DO YOU CHOOSE THE BEST MODEL????

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#create a dictionary 
model_params = {                
    
    "svm": {
         "model": SVC(gamma="auto"),
         "params": {
             "C": [1,10,20],
             "kernel": ["rbf","linear"]
             }
         },
    "random forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [1,5,10]
            }
        },
    "logistic_regression": {
        "model": LogisticRegression(solver="liblinear",multi_class="auto"),
        "params": {
            "C": [1,5,10]
            }
        }
        
 }

#now a for loop
model_params.items()
scores = []

for model_name,mp in model_params.items():
    clf = GridSearchCV(mp["model"],mp["params"],cv=5, return_train_score=False)
    clf.fit(iris.data,iris.target)
    scores.append({
        "model":model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_
        })

df = pd.DataFrame(scores,columns=["model","best_score","best_params"])


#the best model is svm!!!! 
