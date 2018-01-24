import os
import argparse

import numpy as np
from sklearn.preprocessing import normalize


def get_argument_parser(description=None):
    if description is None:
        description = ("This script needs a working path to read store "
                       "trained models.")
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        'path', help='path to the experiment data', default=os.path.curdir)
    return parser


def get_path_from_cli_arguments(description=None):
    return get_argument_parser(description=description).parse_args().path


class BaseModel(object):

    def __init__(self, predictor, context_actions, features="both",
                 randomize_context=None):
        self.model = predictor
        self.actions = context_actions
        self.actions_idx = {a: i for i, a in enumerate(context_actions)}
        self.features = features
        self.randomize_context = randomize_context

    @property
    def n_actions(self):
        return len(self.actions)

    def _transform_labels(self, labels):
        return [self.actions_idx[l] for l in labels]

    def _get_X(self, X_context, X_speech):
        assert(X_context.shape[1] == self.n_actions)
        if self.features == 'context':
            return X_context
        elif self.features == 'speech':
            return X_speech.toarray()
        elif self.features == 'both':
            return np.concatenate([X_context, X_speech.toarray()], axis=1)

    def _predict_proba(self, X_context, X_speech):
        p = np.zeros((X_context.shape[0], self.n_actions))
        p[:, self.model.classes_.tolist()] = self.model.predict_proba(
            self._get_X(X_context, X_speech))
        return p

    def fit(self, X_context, X_speech, labels,
            sample_weight=None, online=False):
        X = self._get_X(X_context, X_speech)
        if self.randomize_context:
            Xc = X_context.copy()
            np.random.shuffle(Xc)
            Xr = self._get_X(Xc, X_speech)
            weights = np.ones((X.shape[0])) if sample_weight is None \
                else sample_weight
            sample_weight = np.concatenate(
                (weights, self.randomize_context * weights))
            X = np.concatenate((X, Xr), axis=0)
            labels = np.concatenate((labels, labels))
        if online:
            lbls = self._transform_labels(labels)
            for i in range(0, X.shape[0]):
                self.model.partial_fit(X[i, :].reshape(1, -1),
                                       [lbls[i]],
                                       sample_weight=[sample_weight[i]],
                                       classes=np.unique(lbls))
        else:
            self.model.fit(X, self._transform_labels(labels),
                           sample_weight=sample_weight)
        return self

    def partial_fit(self, X_context, X_speech, labels, classes):
        X = self._get_X(X_context, X_speech)
        lbls = self._transform_labels(labels)
        clsses = self._transform_labels(classes)
        self.model.partial_fit(X, lbls,
                               classes=clsses)
        return self

    def predict(self, X_context, X_speech, exclude=[]):
        if exclude:
            raise NotImplementedError  # TODO
        return [self.actions[i]
                for i in self.model.predict(self._get_X(X_context, X_speech))]

    @classmethod
    def model_generator(cls, predictor_class, **predictor_args):

        def f(context_actions, **kwargs):
            return cls(predictor_class(**predictor_args), context_actions,
                       **kwargs)

        return f


class ContextFilterModel(BaseModel):

    def predict(self, X_context, X_speech, exclude=[]):
        excl = np.ones(X_context.shape)
        excl[:, [self.actions_idx[a] for a in exclude]] = 0
        probas = self._predict_proba(X_context, X_speech)
        return [self.actions[i] for i in np.argmax(X_context * excl * probas, axis=1)]


class PragmaticModel(ContextFilterModel):

    def __init__(self, predictor, context_actions, features="both",
                 alpha=1., beta=3.):
        super(PragmaticModel, self).__init__(predictor, context_actions,
                                             features=features)
        self.alpha = alpha
        self.beta = beta
        self._p = None
        self._q = None

    def fit(self, X_context, X_speech, labels):
        super(PragmaticModel, self).fit(X_context, X_speech, labels)
        self._p = X_context.sum(axis=0) + self.alpha
        self._p /= self._p.sum()
        probas = super(PragmaticModel, self)._predict_proba(X_context, X_speech)
        self._q = np.power(probas, self.beta).sum(axis=0) + 1.e-8
        return self

    def _predict_proba(self, X_context, X_speech):
        probas = super(PragmaticModel, self)._predict_proba(X_context, X_speech)
        l_probas = (np.power(probas, self.beta) *
                    self._p[None, :] / self._q[None, :])
        return normalize(l_probas, axis=1)
