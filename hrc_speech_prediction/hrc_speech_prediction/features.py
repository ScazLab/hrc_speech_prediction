import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from hrc_speech_prediction.wrappers import Word2Vectorizer

def get_bow_features(data, tfidf=True, n_grams=(1, 1), max_features=None):
    if tfidf:
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = CountVectorizer(ngram_range=n_grams,
                                     max_features=max_features)
    X = vectorizer.fit_transform(data.utterances)
    return X, vectorizer

def get_context_features(data, actions=None):
    all_labels = list(set(data.labels)) if actions is None else actions
    X = np.ones((data.n_samples, len(all_labels)), dtype='bool')
    for p in data.data:
        for trial in data.data[p]:
            trial_ids = trial.ids
            trial_label_ids = [all_labels.index(i) for i in trial.labels]
            for i, l in enumerate(trial_label_ids):
                # mark missing object after it's taken
                X[trial_ids[(i + 1):], l] = 0
    return X, all_labels

def get_w2v_features(data):
    vectorizer = Word2Vectorizer()
    X = vectorizer.fit_transform(data.utterances)
    return X, vectorizer
