from sklearn.naive_bayes import GaussianNB
import os
import numpy as np
from hrc_speech_prediction import (data, features)

class evaluateModel(object):
    def __init__(self, model, data_path):
        """
        Given a model and a path to the data, will run a number of different
        evaluations
        """
        self.data = data.TrainData.load(os.path.join(data_path,"train.json"))

        self.model = model
        self.m_data, _  = features.get_context_features(self.data)
        self.rows,self.columns = self.m_data.shape


    def testOnOneParticipant(self):
        """
        Leaves on participant out of training and then tests on them. Does
        this for each participant
        """

        participants = self.data.participants
        score_avg = 0

        # Get the indices for training and testing data
        for tst in participants:
            

            # Get the labels for the testing participant
            test_Y = list(self.data.data[tst].labels)

            test_idx = list(self.data.data[p].ids)
            test_X = self.m_data[test_idx,:]

            train_idx = [self.m_data.data[p].ids
                         for p in participants if p != tst]
            train_X = self.m_data[i for i in pi for pi in train_idx,:]

            # for train in participants:
            #     if train != tst:
            #         train_ids.append(list(self.data.data[train].ids))
            #         # Get the list of labels (i.e. actions) for a participant
            #         train_Y = np.append(train_Y,
            #                             list(self.data.data[tst].labels),
            #                             axis=0)
            #         print(train_X.shape)
            #         train_X = np.vstack((train_X,
            #                             self.m_data[train_ids,:]))

            model = self.model().fit(train_X, train_Y)
            prediction = self.model.predict(test_X.reshape(1, -1))[0]

            score = sklearn.metrics.accuracy_score(test_Y,
                                                   prediction,
                                                   normalize=True,
                                                   sample_weight=None)
            score_avg += score_avg

            print("Testing on participant {}\n\t Accuracy is: {}\n"
                  .format(tst, score))
                    

        score_avg = score_avg / self.data.n_participants() * 100
        print("On average {}\% of the labels were correctly predicted"
              .format(score_avg))
                    

    def testOnOneTrial(self):
        """
        Excludes on trial from training (i.e. A, B, or C) and uses these
        excluded trials as tests
        """
        trials = ["A", "B", "C"]
        participants = self.data.data.keys()
        score_avg = 0

        for t in trials:
            train_trials = [i for i in trials if i != t]

            train_X = np.array([])
            train_Y = np.array([])

            test_X = np.array([])

            for p in participants:
                sesh = self.data.data[p]

                if t in sesh.order:
                    test_ids = sesh[sesh.order.index(t)].ids
                    test_X = np.append(test_X,
                                self.m_data[test_ids],
                                axis=0)

                    test_Y = np.append(test_Y,
                                       sesh[sesh.order.index(t)].lables,
                                       axis=0)

                train_ids = []
                for i in train_trials:
                    if i in sesh.order:
                        train_ids.append(sesh[sesh.order.index(i)].ids)
                        train_Y = np.append(train_Y,
                                            sesh[sesh.order.index(i)].labels,
                                            axis=0)
            
                train_X = np.append(train_X,
                                    self.m_data(train_ids),
                                    axis=0)

            model = self.model().fit(train_X, train_Y)
            prediction = self.model.predict(test_X.reshape(1, -1))[0]

            score = sklearn.metrics.accuracy_score(test_Y,
                                                   prediction,
                                                   normalize=True,
                                                   sample_weight=None)

            score_avg += score_avg

            print("Testing on trial {} \n\t Accuracy is: {}\n"
                  .format(t, score))
                    

        score_avg = score_avg / 3.0 * 100
        print("On average {}\% of the labels were correctly predicted"
              .format(score_avg))



if __name__ == '__main__':
     path = "/home/scazlab/Desktop/speech_prediction_bags/ExperimentData/"
     ev = evaluateModel(GaussianNB, path)
     ev.testOnOneParticipant()
    # ev.testOnOneParticipant()
