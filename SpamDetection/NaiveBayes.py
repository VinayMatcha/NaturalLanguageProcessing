import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier


data = pd.read_csv("spambase.data").values
np.random.shuffle(data)
X = data[:, :48]
Y = data[:, -1]
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("score using Naive bayes is ", model.score(Xtest, Ytest))

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("score using AdaBoostClassifier is ", model.score(Xtest, Ytest))