import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

from hrc_speech_prediction import context_model


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


class JointModel(object):
    """Uses one model to learn from speech and context features.

    Requires compatible features (i.e. vectorized context).

    When used with features, may only consider speech/context and ignore
    the other.
    """

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


class ContextFilterModel(JointModel):
    def predict(self, X_context, X_speech, exclude=[]):
        excl = np.ones(X_context.shape)
        excl[:, [self.actions_idx[a] for a in exclude]] = 0
        probas = self._predict_proba(X_context, X_speech)
        return [
            self.actions[i]
            for i in np.argmax(X_context * excl * probas, axis=1)
        ]


class PragmaticModel(ContextFilterModel):
    def __init__(self,
                 predictor,
                 context_actions,
                 features="both",
                 alpha=1.,
                 beta=3.):
        super(PragmaticModel, self).__init__(
            predictor, context_actions, features=features)
        self.alpha = alpha
        self.beta = beta
        self._p = None
        self._q = None

    def fit(self, X_context, X_speech, labels):
        super(PragmaticModel, self).fit(X_context, X_speech, labels)
        self._p = X_context.sum(axis=0) + self.alpha
        self._p /= self._p.sum()
        probas = super(PragmaticModel, self)._predict_proba(
            X_context, X_speech)
        self._q = np.power(probas, self.beta).sum(axis=0) + 1.e-8
        return self

    def _predict_proba(self, X_context, X_speech):
        probas = super(PragmaticModel, self)._predict_proba(
            X_context, X_speech)
        l_probas = (
            np.power(probas, self.beta) * self._p[None, :] / self._q[None, :])
        return normalize(l_probas, axis=1)


class CombinedModel(object):
    def __init__(self,
                 vectorizer,
                 model_generator,
                 actions,
                 speech_eps=0.15,
                 context_eps=0.15):

        self._speech_eps = speech_eps
        self._context_eps = context_eps

        self.actions = actions

        self.model_generator = model_generator
        self._vectorizer = vectorizer

        self.context_model = context_model.ContextTreeModel(
            self.actions, eps=context_eps)
        self.speech_model = None

        self._X_context = np.ones((1, len(self.actions)), dtype='bool')

    def fit(self, ctxt, speech, actions):
        self.context_model.fit(ctxt, actions)
        self.speech_model = self.model_generator(
            self.actions, features="speech").fit(
                self._X_context, speech, actions, sample_weight=None)

    def partial_fit(self, ctxt, speech, action):

        if isinstance(speech, str):
            x_u = self._vectorizer.transform([speech])
        else:
            x_u = speech  # Then the input is an numpy array already


        self.context_model.fit(ctxt, action)

        if self.speech_model:
            self.speech_model.partial_fit(self._X_context, x_u, action,
                                          self.actions)
        else:
            self.speech_model = self.model_generator(
                self.actions, features="speech").partial_fit(
                    self._X_context, x_u, action, classes=self.actions)

    def predict(self,
                cntxt,
                utter=None,
                model="both",
                plot=False,
                return_probs=False):

        if model == "context" is not None:
            probs = self.get_context_probs(cntxt)
        elif model == "speech" and utter is not None:
            probs = self.get_speech_probs(utter)
        elif model == "both" and utter is not None:
            context_probs = self.get_context_probs(cntxt)
            speech_probs = self.get_speech_probs(utter)
        else:
            raise "Error, bad inputs to predict()!"

        probs = np.multiply(context_probs, speech_probs)

        action = self.get_probable_action(probs)
        # self.curr = self.curr._get_next_node(action)

        if plot and model == "both":
            self.plot_predicitions(speech_probs, context_probs, probs, utter)
        if return_probs and model == "both":
            return action, speech_probs, context_probs, probs
        else:
            return action, probs

    def get_speech_probs(self, utter):
        "Takes a speech utterance and returns probabilities for each \
            possible action on the speech model alone"

        if isinstance(utter, str):
            x_u = self._vectorizer.transform([utter])
        else:
            x_u = utter  # Then the input is an numpy array already

        probs = self.speech_model._predict_proba(self._X_context, x_u)[0]

        return self._apply_eps(self._speech_eps, probs)

    def get_context_probs(self, cntxt):
        curr = self.context_model.curr(cntxt)
        probs = curr._get_context_probs(self._context_eps, self.actions)

        return self._apply_eps(self._context_eps, probs)

    def _apply_eps(self, eps, p):
        u = np.array([1.0 / len(self.actions) for i in self.actions])

        return (1.0 - eps) * p + (u * eps)

    def get_probable_action(self, probs):
        return self.actions[np.argmax(probs)]

    def plot_predicitions(self,
                          speech,
                          context,
                          both,
                          utter,
                          actual=None,
                          save_path=None):
        "Plots the probabilities for each possible action provided by speech, \
        context, and speech + context "

        X = np.arange(len(both))
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Want to normalize 'both' probs for easier visual comparison
        nrmlz = 1.0 / sum(both)

        ax.bar(X - 0.2, speech, width=0.2, color='r', align='center')
        ax.bar(X, context, width=0.2, color='b', align='center')
        ax.bar(X + 0.2, both * nrmlz, width=0.2, color='g', align='center')

        ax.legend(('Speech', 'Context', 'Both'))

        rects = ax.patches
        max_prob = max(both * nrmlz)

        # This draws a star above most probable action
        for r in rects:
            if r.get_height() == max_prob:
                ax.text(
                    r.get_x() + r.get_width() / 2,
                    r.get_height() * 1.01,
                    '*',
                    ha='center',
                    va='bottom')

        if actual:
            ax.text(self.speech_model.actions.index(actual), max_prob, "$")

        plt.xticks(X, self.speech_model.actions, rotation=70)
        plt.title(utter)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show(block=False)

    def __str__(self):
        return self.context_model.__str__()
