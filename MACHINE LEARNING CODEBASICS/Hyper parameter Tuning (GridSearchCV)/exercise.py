import numpy as np
import pandas as pd 
from sklearn.datasets import load_digits

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
digits= load_digits()
model_params = {
    "svm": {
        "model": SVC(gamma="auto"),
        "params": {
            "C": [1,10,20],
            "kernel": ["rbf","linear"]
            }
        },
    "random_forest": {
    "model": RandomForestClassifier(),
    "params": { 
        "n_estimators": [1,5,10]
             }
        },
    "logistic_regression":{
        "model": LogisticRegression(solver="liblinear",multi_class="auto"),
        "params": {
            "C": [1,5,10]}
        },
    "naive_bayes_gaussian": {
        "model": GaussianNB(),
        "params": {}
        },
    "naive_bayes_multinomial": {
        "model": MultinomialNB(),
        "params": {}
        },
    "decision_tree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "criterion": ["gini","entropy"],
            }
        }
    }

from sklearn.model_selection import GridSearchCV
import pandas as pd

scores = []

#for loop


for model_name,mp in model_params.items():
    clf = GridSearchCV(mp["model"],mp["params"],cv = 5,return_train_score=False)
    clf.fit(digits.data,digits.target)
    scores.append({
        "model": model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_})

df = pd.DataFrame(scores,columns=["model","best_score","best_params"])


#svm is the best model!!!



















