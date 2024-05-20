from profanityfilter import ProfanityFilter #it's a library for detecting proform word in any given list
def check(a):
    import string
    print(string.punctuation)
    import re
    import nltk
    nltk.download("stopwords")
    nltk.download("wordnet")
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
tweet="!!! RT @mayasolovely: As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out..."
k = check(tweet)
pf = ProfanityFilter()
Approval=pf.is_profane(k)
if Approval==True:
    print("spam contains")
