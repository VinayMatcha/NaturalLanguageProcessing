import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()
titles = [t.rstrip() for t in open('all_book_titles.txt')]
stopwords = set(t.rstrip() for t in open('stopwords.txt'))
stopwords = stopwords.union({
'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth',
})



def tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens


word_index_map = {}
currentIndex = 0
books_tokenized = []
book_titles = []
index_to_word = []

for title in titles:
    try:
        title = title.encode('ascii', 'ignore').decode('utf-8')
        book_titles.append(title)
        tokens = tokenizer(title)
        books_tokenized.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = currentIndex
                currentIndex += 1
                index_to_word.append(token)
    except:
        pass

def tokensToVector(tokens):
    X = np.zeros(len(word_index_map))
    for token in tokens:
        i = word_index_map[token]
        X[i] = 1
    return X

N = len(books_tokenized)
D= len(word_index_map)
X = np.zeros((N, D))
i = 0
for tokens in books_tokenized:
    X[i:, ] = tokensToVector(tokens)
    i += 1

'''because we are calculating document wise we need transpose'''
X = X.T
svd = TruncatedSVD()
Z = svd.fit_transform(X)
plt.scatter(Z[:, 0], Z[:, 1])
for i in range(D):
    plt.annotate(text=index_to_word[i], xy=(Z[i, 0], Z[i, 1]))
plt.show()
plt.savefig("images/lsa_on_booktitles")



