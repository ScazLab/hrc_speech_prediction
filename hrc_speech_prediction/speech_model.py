import numpy as np


class SpeechModel(object):
    def __init__(self,
                 predictor,
                 context_actions,
                 features="both",
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
        assert (X_context.shape[1] == self.n_actions)
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

    def fit(self,
            X_context,
            X_speech,
            labels,
            sample_weight=None,
            online=False):
        X = self._get_X(X_context, X_speech)
        if self.randomize_context:
            Xc = X_context.copy()
            np.random.shuffle(Xc)
            Xr = self._get_X(Xc, X_speech)
            weights = np.ones((X.shape[0])) if sample_weight is None \
                else sample_weight
            sample_weight = np.concatenate((weights,
                                            self.randomize_context * weights))
            X = np.concatenate((X, Xr), axis=0)
            labels = np.concatenate((labels, labels))
        if online:
            lbls = self._transform_labels(labels)
            for i in range(0, X.shape[0]):
                self.model.partial_fit(
                    X[i, :].reshape(1, -1), [lbls[i]],
                    sample_weight=[sample_weight[i]],
                    classes=np.unique(lbls))
        else:
            self.model.fit(
                X, self._transform_labels(labels), sample_weight=sample_weight)
        return self

    def partial_fit(self, X_context, X_speech, labels, classes=None):
        X = self._get_X(X_context, X_speech)
        lbls = self._transform_labels(labels)
        clsses = np.array(self._transform_labels(classes))
        self.model.partial_fit(X, lbls, classes=clsses)
        return self

    def predict(self, X_context, X_speech, exclude=[]):
        if exclude:
            raise NotImplementedError  # TODO
        return [
            self.actions[i]
            for i in self.model.predict(self._get_X(X_context, X_speech))
        ]

    @classmethod
    def model_generator(cls, predictor_class, **predictor_args):
        def f(context_actions, **kwargs):
            return cls(
                predictor_class(**predictor_args), context_actions, **kwargs)

        return f
