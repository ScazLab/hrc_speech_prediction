"""
Wrappers

- for gensim word2vec impelementation
so that it can be used side by side with
scikit vectorizers

functions to implement: (functions that I found used in the hrc code)
vectorizer.transform([speech])
vectorizer.transfrom([utter])
vectorizer.fit_transform(data.utterances)
"""

import numpy as np
from gensim.models import word2vec


class Word2Vectorizer:
    def fit(sentences):
        self.model = word2vec.Word2Vec(sentences)
        return self.model

    def transform(sentences):
        fit(sentences)
        return self.model.syn0

    def fit_transform(sentences):
        fit(sentences)
        return transform()
