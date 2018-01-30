import os.path
from unittest import TestCase

from sklearn.linear_model import SGDClassifier

from hrc_speech_prediction import context_model
from hrc_speech_prediction.data import (ALL_ACTIONS, TRAIN_PARTICIPANTS,
                                        TrainData)
from hrc_speech_prediction.defaults import MODEL_PATH
from hrc_speech_prediction.features import get_bow_features
from hrc_speech_prediction.models import CombinedModel
from hrc_speech_prediction.speech_model import SpeechModel


class TestContextTreeModel(TestCase):
    def setUp(self):
        self.model = context_model.ContextTreeModel(ALL_ACTIONS)

        self.model.fit([[], [], ["a"], ["z", "b", "c"]], ["a", "a", "b", "d"])

    def test_n_children(self):
        self.assertEqual(self.model.root.n_children, 2)


class TestCombinedModel(TestCase):
    def setUp(self):
        data_path = (os.path.join(
            os.path.dirname(__file__), "test_participant.json"))

        TFIDF = False
        N_GRAMS = (1, 2)

        self.data = TrainData.load(data_path)

        train_ids = [t.ids for t in self.data.data["1.ABC"]]
        flat_train_ids = list(self.data.data["1.ABC"].ids)
        self.train_cntxt, self.train_acts = self._format_cntxt_indices(
            train_ids)

        self.X_speech, vectorizer = get_bow_features(
            self.data, tfidf=TFIDF, n_grams=N_GRAMS, max_features=None)
        self.X_speech = self.X_speech[flat_train_ids, :]

        model_gen = SpeechModel.model_generator(
            SGDClassifier,
            loss='log',
            average=True,
            penalty='l2',
            alpha=0.0002)

        self.combined_model = CombinedModel(
            vectorizer,
            model_gen,
            ALL_ACTIONS,
            speech_eps=0.15,
            context_eps=0.15)
        # self.combined_model.add_branch(["foot_2"])
        # self.combined_model.add_branch(["top_1"])
        # self.combined_model.add_branch(["foot_2", "foot_1", "leg_1"])
        # self.combined_model.add_branch(["chair_back", "seat", "back_1"])

        self.test_utter = "The green piece with two black stripes"
        self.test_cntxt = []

    def test_n_children(self):
        self.combined_model.fit(self.train_cntxt, self.X_speech,
                                self.train_acts)
        self.assertEqual(self.combined_model.context_model.root.n_children, 2)

    def test_predict(self):
        self.combined_model.fit(self.train_cntxt, self.X_speech,
                                self.train_acts)
        act, probs = self.combined_model.predict(self.test_cntxt,
                                                 self.test_utter)
        self.assert_("foot" in act)

    def test_prob_normilization(self):
        self.combined_model.fit(self.train_cntxt, self.X_speech,
                                self.train_acts)
        c_probs = self.combined_model.get_context_probs(self.test_cntxt)
        s_probs = self.combined_model.get_speech_probs(self.test_utter)

        self.assertAlmostEqual(sum(c_probs), 1.0, places=4)
        self.assertAlmostEqual(sum(s_probs), 1.0, places=4)

    def test_incremental_learning(self):
        train_ctxt_1 = [[]]
        train_ctxt_2 = [[]]

        train_utter_1 = "blue piece with two white stripes"
        train_utter_2 = "I am feeling fat and sassy"

        act_1 = ["top_2"]
        act_2 = ["top_1"]

        self.combined_model.partial_fit(train_ctxt_1, train_utter_1, act_1)
        self.combined_model.partial_fit(train_ctxt_2, train_utter_2, act_2)

        self.assertEqual(self.combined_model.context_model.root.n_children, 2)

    def _format_cntxt_indices(self, indices):
        cntxts = []
        actions = []

        all_labels = list(self.data.labels)
        for t in indices:
            cntxt = []
            for i in t:
                label = all_labels[i]
                cntxts.append([c for c in cntxt])
                actions.append(label)

                cntxt.append(label)

        return cntxts, actions
