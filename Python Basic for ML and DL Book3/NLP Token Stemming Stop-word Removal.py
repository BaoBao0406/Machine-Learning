
# Tokenize
def tokenizer(text):
    return text.split()
print(tokenizer('runners like running and thus they run'))


# Stemming
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
print(tokenizer_porter('runners like running and thus they run'))


# Stop-word removal
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop ])
