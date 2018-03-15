
# coding: utf-8


import nltk
from collections import Counter
import pandas as pd
import string
import numpy as np
import sklearn
import time

def process(docs_ls):
    docs_raw = [tokenize(doc) for doc in docs_ls]
    docs = remove_stopwords(docs_raw)
    return docs

def remove_stopwords(docs):
    stopwords=nltk.corpus.stopwords.words('english')
    stopwords = tokenize(' '.join(stopwords))
    stopwords.extend(get_rare_words(docs))
    stopwords = set(stopwords)
    res = [[word for word in doc if word not in stopwords ] for doc in docs]
    return res


def tokenize(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    text = text.lower()
    text = text.replace("'s", '')
    text = text.replace("'", '')
    
    punc = string.punctuation
    for c in punc:
        if c in text:
            text = text.replace(c, ' ')
    
    tokens = nltk.word_tokenize(text)
#     print(tokens)
    res = []
    
    for token in tokens:
        try:
            word = lemmatizer.lemmatize(token)
            if len(word)>1:
                try:
                    int(word)
                except:
                    res.append(str(word))
        except:
            continue
    docs=nltk.word_tokenize(" ".join(res))
    return res


def get_rare_words(tokens_ls):
    """ use the word count information across all tweets in training data to come up with a feature list
    Inputs:
        processed_tweets: pd.DataFrame: the output of process_all() function
    Outputs:
        list(str): list of rare words, sorted alphabetically.
    """
    counter = Counter([])
    for tokens in tokens_ls:
        counter.update(tokens)
    
    rare_tokes = [k for k,v in counter.items() if v==1]
    rare_tokes.sort()
    return rare_tokes


