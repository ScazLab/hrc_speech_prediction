from sklearn.externals import joblib

from unittest import TestCase

from hrc_speech_prediction import combined_model as cm
from hrc_speech_prediction.defaults import MODEL_PATH


class TestCombinedModel(TestCase):

    def setUp(self):
        model_path = MODEL_PATH

        self.speech_model = joblib.load(model_path + "combined_model_0.150.15.pkl")
        self.vectorizer = joblib.load(model_path + "vocabulary.pkl")

        self.combined_model = cm.CombinedModel(speech_model=self.speech_model,
                                               root=cm.Node(),
                                               vectorizer=self.vectorizer)

        self.combined_model.add_branch(["foot_2"])
        self.combined_model.add_branch(["top_1"])
        self.combined_model.add_branch(["foot_2", "foot_1", "leg_1"])
        self.combined_model.add_branch(["chair_back", "seat", "back_1"])

        self.test_utter = "Pass me the blue piece with two red stripes"

    def test_n_children(self):
        print("test_n_children()")
        self.assertEqual(self.combined_model.root.n_children, 3)

    def test_prob_normilization(self):
        c_probs = self.combined_model.get_context_probs()
        s_probs = self.combined_model.get_speech_probs(self.test_utter)

        print(self.combined_model)
        self.assertAlmostEqual(sum(c_probs), 1.0, places=4)
        self.assertAlmostEqual(sum(s_probs), 1.0, places=4)
