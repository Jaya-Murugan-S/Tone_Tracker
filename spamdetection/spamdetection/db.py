from profanityfilter import ProfanityFilter #it's a library for detecting proform word in any given list
import pandas as panda
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import nltk
def single_comment(tweet):
    def data(a):
        import re
        e=[]
        nltk.download("omw-1.4")
        ps=nltk.WordNetLemmatizer()
        ae=nltk.PorterStemmer()
        stopwords=nltk.corpus.stopwords.words("english")
        text="".join([word for word in a if word not in string.punctuation])
        text=[word for word in text.split() if word not in stopwords]
        text=[ps.lemmatize(word) for word in text]
        text=" ".join([ae.stem(word) for word in text])
        return text

    
    tweet=data(tweet)
    print(tweet)
    pf = ProfanityFilter()
    Approval=pf.is_profane(tweet)
    if Approval == True:
        return "s"
    else:
        return "no"
