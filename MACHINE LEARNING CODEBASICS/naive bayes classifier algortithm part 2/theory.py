import numpy as np
import pandas as pd
##################################
#email spam detection!
##################################



df = pd.read_csv("spam.csv")
df.head()

df.groupby("Category").describe() #4825 ham and 747 are spams
#lambda is a function 

df["spam"] = df["Category"].apply(lambda x: 1 if x=="spam" else 0) #1 == "spam"
df.head()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df.Message,df.spam,test_size=0.25)

#the message column is stil a text we need to handle that 
#so we use
#count vectorizer technique

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()
#now text in convert into numbers

#remember we have 3 type of bayes, bernoulli naive bayes, multinomial naive bayes and gaussian naive bayes
#we are using multinomial naive bayes

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)

#faccio un esempio

emails = ["hey mohan, can we get together to watch football game tomorrow?",
          "upto 20% discount on parking, exclusive offer just for you. don't miss this reward!"]
#the first one should be a ham, the second one a spam
#let's see if the model predict right

emails_count = v.transform(emails)
model.predict(emails_count) #infatti è giusto


X_test_count=v.transform(X_test)
model.score(X_test_count,y_test) #really precise

#for email spam the naive model works really well

#another way to do it with pipeline
#cosi non devo ogni volta trasformare il testo in vettore
#ed è molto piu semplice

from sklearn.pipeline import Pipeline
clf = Pipeline([
    ("vectorizer",CountVectorizer()), #first step convert text into vector
    ("nb",MultinomialNB()) #then apply multinomial naive bayes
    ])

clf.fit(X_train,y_train)

clf.score(X_test,y_test) #perfetto








