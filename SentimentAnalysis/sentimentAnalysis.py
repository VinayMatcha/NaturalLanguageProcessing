import numpy as np
import pandas as pd
import nltk
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.utils import shuffle


wordnet_lemmatizer = WordNetLemmatizer()
wordCount = 0
'''map for storing the words and indexes'''
word_index_map = {}
current_index = 0
positiveIndexed = []
negativeIndexed = []
totalReviews = []

def tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens




'''contains all the words which doesn't have effect on reviews'''

stopwords = set(w.rstrip() for w in open('stopwords.txt'))
positiveReviews = BeautifulSoup(open('../electronics/positive.review').read(),features="html.parser")
negativeReviews = BeautifulSoup(open('../electronics/negative.review').read(), features="html.parser")
positiveReviews = positiveReviews.find_all('review_text')
negativeReviews = negativeReviews.find_all('review_text')

''' print(len(negativeReviews), " ", len(positiveReviews))
 making the positive and negative reviews count same'''

diff = len(positiveReviews) - len(negativeReviews)
idx = np.random.choice(len(negativeReviews), size = diff)
extraNegativeReviews = [negativeReviews[i] for i in idx]
negativeReviews = negativeReviews + extraNegativeReviews

# print(len(negativeReviews), " ", len(positiveReviews))

for review in positiveReviews:
    tokens = tokenizer(review.text)
    totalReviews.append(review.text)
    positiveIndexed.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


for review in negativeReviews:
    tokens = tokenizer(review.text)
    totalReviews.append(review.text)
    negativeIndexed.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


print("total number of diff words ", len(word_index_map))

def tokensToVector(tokens, label):
    X = np.zeros(len(word_index_map)+1)
    for token in tokens:
        i = word_index_map[token]
        X[i] += 1
    X = X/X.sum()
    X[-1] = label
    return X




ReviewCount = len(totalReviews)
data = np.zeros((ReviewCount, len(word_index_map)+1))
i = 0
for tokens in positiveIndexed:
    data[i:, ] = tokensToVector(tokens, 1)
    i += 1

for tokens in negativeIndexed:
    data[i:, ] = tokensToVector(tokens, 0)
    i += 1

totalReviews, data = shuffle(totalReviews, data)
X = data[:, :-1]
Y = data[:, -1]
Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]
Xtest = X[-100:, ]
Ytest = Y[-100:, ]


model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Training accuracy is ", model.score(Xtest, Ytest))
print("Score for the reviews is ", model.score(Xtest, Ytest))

threshold = 0.5
for word, i in word_index_map.items():
    weight = model.coef_[0][i]
    if(weight > threshold or weight < -threshold):
        print(word, " ", weight)

