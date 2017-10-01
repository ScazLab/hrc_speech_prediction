from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
from hrc_speech_prediction import (data, features)

class evaluateModel(object):
    def __init__(self, model, data_path, **kwargs):
        """
        Given a model and a path to the data, will run a number of different
        evaluations
        """
        self.data = data.TrainData.load(os.path.join(data_path,"train.json"))
        self.model = model
        self.args = kwargs
        self.m_features, _ = features.get_context_features(self.data)
        self.m_speech, _  = features.get_bow_features(self.data)
        self.m_all = np.concatenate((self.m_features,
                                     self.m_speech.toarray()),
                                    axis=1)


    def testOnOneParticipant(self,data_type="context"):
        """
        Leaves on participant out of training and then tests on them. Does
        this for each participant
        """

        print("Running test on one participant...")

        participants = self.data.participants
        results = {}
        score_avg = 0

        if data_type is "context":
            m_data = self.m_features
        elif data_type is "speech":
            m_data = self.m_speech.toarray()
        else:
            m_data = self.m_all

        # Get the indices for training and testing data
        for tst in participants:
            

            # Get the labels for the testing participant
            test_Y = list(self.data.data[tst].labels)

            test_idx = list(self.data.data[tst].ids)
            test_X = m_data[test_idx, :]

            train_idx = [list(self.data.data[p].ids)
                         for p in participants if not p == tst]
            train_X = m_data[[i for pi in train_idx for i in pi], :]
            train_Y = [list(self.data.labels)[i]
                       for pi in train_idx
                       for i in pi]


            model = self.model().fit(train_X, train_Y)
            prediction = model.predict(test_X)

            #.reshape(1, -1
            score = metrics.accuracy_score(test_Y,
                                           prediction,
                                           normalize=True,
                                           sample_weight=None)
            score_avg += score

            results[tst] = score 

        score_avg = score_avg / self.data.n_participants 

        print("----------Testing on {} ---------".format(data))
        print("{:<15} {:<10}".format('Pariticpant Tested', 'Accuracy'))
        for k,v in results.iteritems():
            print("{:<15} {:<10}".format(k,v))

        print("Average {}\n"
              .format(score_avg))

    def testOnOneTrial(self, data_type="context"):
        """
        Excludes on trial from training (i.e. A, B, or C) and uses these
        excluded trials as tests
        """
        print("Running test on one trial...")

        trials = ['A', 'B', 'C']
        participants = self.data.participants
        score_avg = 0

        results = {}

        if data_type is "context":
            m_data = self.m_features
        elif data_type is "speech":
            m_data = self.m_speech.toarray()
        else:
            m_data = self.m_all

        for t in trials:
            train_trials = [i for i in trials if i != t]

            test_idx = [i
                        for part in self.data.data
                        for trial in self.data.data[part]
                        for i in trial.ids
                        if t == trial.instruction]

            train_idx = [i for i in range(0, self.data.n_samples)
                         if i not in test_idx]

            test_X = m_data[test_idx, :]
            train_X = m_data[train_idx, :]

            train_Y = [list(self.data.labels)[i]
                       for i in train_idx]
            test_Y = [list(self.data.labels)[i]
                       for i in test_idx]


            model = self.model(**self.args).fit(train_X, train_Y)
            prediction = model.predict(test_X)

            score = metrics.accuracy_score(test_Y,
                                           prediction,
                                           normalize=True,
                                           sample_weight=None)

            score_avg += score
            results[t] = score
            # print("Testing on trial {} \n\t Accuracy is: {}\n"
            #       .format(t, score))
                    

        print("----------testing on {} ---------".format(data))
        print("{:<15} {:<10}".format('Trial Tested','Accuracy'))
        for k,v in results.iteritems():
            print("{:<15} {:<10}".format(k,v))

        score_avg = score_avg / 3.0 
        print("Average {}\n"
              .format(score_avg))

    def cross_validation(self, data_type="context"):

        """
        Exclude one sample from training and test on it.
        """
        print("Running cross validation...")

        results = np.empty((1,))

        if data_type is "context":
            m_data = self.m_features
        elif data_type is "speech":
            m_data = self.m_speech.toarray()
        else:
            m_data = self.m_all

        for i in range (0, self.data.n_samples):
            train_idx = [j for j in range(0, self.data.n_samples)
                         if not j == i]

            train_X = m_data[train_idx, :]
            test_X = m_data[i, :]

            train_Y = [list(self.data.labels)[j] for j in train_idx]
            test_Y = [list(self.data.labels)[i]]

            model = self.model(**self.args).fit(train_X, train_Y)
            prediction = model.predict(test_X.reshape(1, -1))

            score = metrics.accuracy_score(test_Y,
                                           prediction,
                                           normalize=True,
                                           sample_weight=None)

            results = np.append(results, score)

        print("----------testing on {} ---------".format(data))
        print("Avg: {}, std dev: {}\n".format(np.mean(results),
                                              np.std(results)))


if __name__ == '__main__':
     #path = "/home/scazlab/Desktop/speech_prediction_bags/ExperimentData/"
    path = "/home/ros/ros_ws/src/hrc_speech_prediction/"
    ev = evaluateModel(KNeighborsClassifier, path, n_neighbors=1)
    ev.cross_validation(data_type="speech")
    # ev.testOnOneParticipant()
