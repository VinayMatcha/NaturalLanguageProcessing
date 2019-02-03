import nltk
import numpy as np
import random

from bs4 import BeautifulSoup

positive_reviews = BeautifulSoup(open('../electronics/positive.review').read(), features='html.parser')
negative_reviews = BeautifulSoup(open('../electronics/negative.review').read(), features='html.parser')
positive_reviews = positive_reviews.find_all('review_text')
negative_reviews = negative_reviews.find_all('review_text')

trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens)-2):
        key = (tokens[i], tokens[i+2])
        if key not in trigrams:
            trigrams[key] = []
        trigrams[key].append(tokens[i+1])


'''Converting these trigrams into probabilities'''
trigram_probabilities = {}
for key, words in trigrams.items():
    if(len(words)>1):
        d = {}
        n = 0
        for word in words:
            if word not in d:
                d[word] = 0
            d[word] += 1
            n += 1
        for word, count in d.items():
            d[word] = float(count)/n
        trigram_probabilities[key] = d

def random_sample(d):
    r = random.random()
    cummulative = 0
    for word, prob in d.items():
        cummulative += prob
        if prob < cummulative:
            return word


def spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print("originla review is ", s)
    tokens = nltk.tokenize.word_tokenize(s)
    for token in tokens:
        r = random.random()
        if r < 0.2:
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                word = random_sample(trigram_probabilities[k])
                tokens[i+1] = word
    print("after changing review is")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))


spinner()