import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("salaries.csv")
df.head(10)

#salary more than 100k or less? let's see

#IT MATTERS IN WHICH ORDER YOU SPLIT THE TREE! -> IT IS GOING TO HAVE AN IMPACT ON THE SCORE

#HOW DO YOU SELECT THE ORDER OF THE FEATURES? 

#the entropy! 

inputs = df.drop("salary_more_then_100k",axis=1)
target = df["salary_more_then_100k"]

#convertiamo le colonne in numeri! 


from sklearn.preprocessing import LabelEncoder

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs ["company_n"] = le_company.fit_transform(inputs["company"])
inputs ["job_n"] = le_company.fit_transform(inputs["job"])
inputs ["degree_n"] = le_company.fit_transform(inputs["degree"])


inputs_n = inputs.drop(["company","job","degree"],axis=1)



from sklearn import tree


model = tree.DecisionTreeClassifier


model.fit(inputs_n,target)












