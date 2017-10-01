import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def get_bow_features(data, use_idf=True):
    vectorizer = TfidfVectorizer(use_idf=use_idf)
    X = vectorizer.fit_transform(data.utterances)
    return X, vectorizer


def get_context_features(data):
    all_labels = list(set(data.labels))
    X = np.ones((data.n_samples, len(all_labels)), dtype='bool')
    for p in data.data:
        for trial in data.data[p]:
            trial_ids = trial.ids
            trial_label_ids = [all_labels.index(i) for i in trial.labels]
            for i, l in enumerate(trial_label_ids):
                # mark missing object after it's taken
                X[trial_ids[(i + 1):], l] = 0
    return X, all_labels
