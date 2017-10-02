import numpy as np


class BaseModel(object):

    def __init__(self, predictor, context_actions, features="both"):
        self.model = predictor
        self.actions = context_actions
        self.actions_idx = {a: i for i, a in enumerate(context_actions)}
        self.features = features

    def _get_X(self, X_context, X_speech):
        assert(X_context.shape[1] == len(self.actions))
        if self.features == 'context':
            return X_context
        elif self.features == 'speech':
            return X_speech.toarray()
        elif self.features == 'both':
            return np.concatenate([X_context, X_speech.toarray()], axis=1)

    def fit(self, X_context, X_speech, labels):
        X = self._get_X(X_context, X_speech)
        self.model.fit(X, [self.actions_idx[l] for l in labels])
        return self

    def predict(self, X_context, X_speech):
        return [self.actions[i]
                for i in self.model.predict(self._get_X(X_context, X_speech))]

    @classmethod
    def model_generator(cls, predictor_class, **predictor_args):

        def f(context_actions, **kwargs):
            return cls(predictor_class(**predictor_args), context_actions,
                       **kwargs)

        return f
