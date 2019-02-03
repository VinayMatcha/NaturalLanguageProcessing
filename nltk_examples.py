import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import wordnet


nltk.pos_tag("vinay is a good boy".split())
porterStemmer = PorterStemmer()
porterStemmer.stem("wolves")
limitizer = wordnet.WordNetLemmatizer()
limitizer.lemmatize("wolves")

sentence = "vinay is born on may 5th 1995"
tags = nltk.pos_tag(sentence.split())
print(tags)
nltk.ne_chunk(tags).draw()


sentence = "i am the ceo of vinay"
tags = nltk.pos_tag(sentence.split())
print(tags)
print(nltk.ne_chunk(tags))
nltk.ne_chunk_sents(tags).draw()