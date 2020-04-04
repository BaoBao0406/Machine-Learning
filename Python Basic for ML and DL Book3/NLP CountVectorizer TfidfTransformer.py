import numpy as np


# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# 1-gram model
count = CountVectorizer()
# 2-gram model
#count = CountVectorizer(ngram_range=(2,2))
docs = np.array(['The sun is shining', 
                 'The weather is sweet', 
                 'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

print(count.vocabulary_)

print(bag.toarray())


# TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
