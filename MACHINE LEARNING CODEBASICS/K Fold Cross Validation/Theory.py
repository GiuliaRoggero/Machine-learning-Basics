#which model is the best to use?
#cross validation to evaluate a model performance

#OPTION 1: use all available data for training and test on same dataset
#it's not very effective! basta cambiare i dati e predice male

#OPTION 2: split available dataset into training and test datasets
#one problem of this approach : magari si allena su il training ma il test è piuttosto diverso

#OPTION 3: K FOLD CROSS VALIDATION
#take the average score of every folds! this technique is very very good

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

digits = load_digits()

df = pd.DataFrame(digits.data)
target = pd.Series(digits.target)

X_train, X_test, y_train,y_test = train_test_split(df,target,test_size=0.3)

len(X_train)
len(X_test)

lr = LogisticRegression()
lr.fit(X_train,y_train)
scorelr = lr.score(X_test,y_test)

svm = SVC()
svm.fit(X_train,y_train)
scoresvm = svm.score(X_test,y_test)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
scorerf = rf.score(X_test,y_test)

#ogni volta con il train_test_split il campione cambia e cambiano anche i risultati
#K fold cross validation rimuove questo problema

from sklearn.model_selection import KFold

kf = KFold(n_splits=3) #n_splits è il numero di cartelle/folds



for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index,test_index)


def get_Score(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)
#modo piu veloce per avere lo score di un modello
get_Score(LogisticRegression(),X_train,X_test,y_train,y_test)

from sklearn.model_selection import StratifiedKFold
fold = StratifiedKFold(n_splits=3)

#Kfold e stratifiedkfold sono praticamente la stessa cosa
scores_l = []
scores_svm = []
scores_rf = []

for train_index,test_index in kf.split(digits.data):
    X_train,X_test,y_train,y_test = digits.data[train_index],digits.data[test_index], \
                                    digits.target[train_index],digits.target[test_index]
    scores_l.append((get_Score(LogisticRegression(),X_train,X_test,y_train,y_test)))                             
    scores_svm.append((get_Score(SVC(),X_train,X_test,y_train,y_test)))
    scores_rf.append((get_Score(RandomForestClassifier(n_estimators=40),X_train,X_test,y_train,y_test)))


scores_l

scores_svm

scores_rf


#ora pero posso usare un solo comando per fare tutto, molto più semplicemente

from sklearn.model_selection import cross_val_score

cross_val_score(LogisticRegression(),digits.data,digits.target)

cross_val_score(SVC(),digits.data,digits.target)

cross_val_score(RandomForestClassifier(),digits.data,digits.target)

#dopo uso np.average() per prendere la media!































