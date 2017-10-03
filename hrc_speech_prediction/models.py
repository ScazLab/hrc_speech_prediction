import numpy as np
from sklearn.preprocessing import normalize


class BaseModel(object):

    def __init__(self, predictor, context_actions, features="both"):
        self.model = predictor
        self.actions = context_actions
        self.actions_idx = {a: i for i, a in enumerate(context_actions)}
        self.features = features

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
                                              self._get_X(X_context,
                                                          X_speech))
        return p

    def fit(self, X_context, X_speech, labels, lam=1):
        # Shuffles the columns of X_context 
        if self.features == 'both':
            X_fake_context = X_context.copy()
            np.random.shuffle(X_fake_context.T)

            X_new_context = np.vstack((X_context, X_fake_context))
            X_new_speech = np.vstack((X_speech, X_speech))
            X = self._get_X(X_new_context, X_new_speech)

            s_weights = np.array([lam if i < self.n_actions else 1.0
                              for i in range(0, X.shape[1])])
            self.model.fit(X,
                       self._transform_labels(labels),
                       sample_weight=s_weights)
        else:
            X = self._get_X(X_context, X_speech)
            self.model.fit(X, self._transform_labels(labels))
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
