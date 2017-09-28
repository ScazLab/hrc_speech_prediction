#!/usr/bin/env python

import logging
import rospy
import rosbag
import sys
import os
import collections as c
import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('This is a log message.')

BAG_DIR_PATH = sys.argv[1]

MODEL_PATH = "../models/speech_model.pkl"
VOCAB_PATH = "../models/vocab.pkl"

# Ros topics we have recorded
left_aruco_topic = '/baxter_aruco_left/markers'
right_aruco_topic = '/baxter_aruco_right/markers'
left_state_topic = '/action_provider/left/state'
right_state_topic = '/action_provider/right/state'
speech_topic = '/speech_to_text/log'
web_topic = '/web_interface/log'

right_obj_dict = c.OrderedDict([(10, 0), (11, 0), (12, 0),
                                (13, 0), (14, 0), (15, 0),
                                (16, 0), (17, 0), (18, 0),
                                (19, 0), (20, 0), (21, 0),
                                (22, 0), (23, 0)])

left_obj_dict = c.OrderedDict([(150, 0), (151, 0), (152, 0),
                                (153, 0), (154, 0), (155,0),
                               (156, 0), (198, 0), (200, 0)])

id_dict = {"seat"          : 200,
           "chair_back"    : 198,
           "leg_1"         : 150,
           "leg_2"         : 151,
           "leg_3"         : 152,
           "leg_4"         : 153,
           "leg_5"         : 154,
           "leg_6"         : 155,
           "leg_7"         : 156,
           "foot_1"        : 10,
           "foot_2"        : 11,
           "foot_3"        : 12,
           "foot_4"        : 13,
           "front_1"       : 14,
           "front_2"       : 15,
           "top_1"         : 16,
           "top_2"         : 17,
           "back_1"        : 18,
           "back_2"        : 19,
           "screwdriver_1" : 20,
           "front_3"       : 22,
           "front_4"       : 23,
           }


def get_time_btw_state(arm_topic, bag_path):
    """returns list of approximate timeframes of when robots arms
    are in the home position (between state changes) and
    the action that triggered the state change """
    bag = rosbag.Bag(bag_path)
    states = []
    avg_diff_list = []
    t_frame = ()
    last_t = ()
    s = False
    # Between START and WORKING states robot is in home position
    for topic, msg, t in bag.read_messages(topics=[arm_topic]):
        if (msg.state == 'START' or msg.state == 'DONE' or
                msg.state == 'ERROR') and not s:
            t_frame += (t, )
            s = True
        elif s and msg.state == 'WORKING':
            t_frame += (t, )
            states.append((t_frame, msg.action + ':' + msg.object))
            avg_diff_list.append(t_frame[1] - t_frame[0])

            s = False
            t_frame = ()

        # THIS is will be appended as a singleton
        # This denotes an arm has reached its end state
        last_t = (t, )

    # We say that the length of the end states last  on average
    # as long as the previous states
    avg_diff = sum([i.to_sec() for i in avg_diff_list]) / len(avg_diff_list)
    avg_diff = rospy.Duration.from_sec(avg_diff)
    states.append((last_t + (avg_diff + last_t[0], ),
                   msg.action + ':' + msg.object))
    bag.close()
    return states


def get_env_vec_for_arm(arm_topic, obj_dict, bag_path):
    """Will construct a state vector for a given arm using timing info
from above function"""
    bag = rosbag.Bag(bag_path)
    obj_dict = obj_dict.copy()
    # will store the vectors corresponding to the state of
    # the world after each action
    vec_of_vecs = []

    # A list of tuples, where each tuple corresponds
    # to the time frames
    t_frames = [t[0] for t in get_time_btw_state(arm_topic, bag_path)]
    t_frame = t_frames.pop(0)

    total_msgs = 0.

    if arm_topic == right_state_topic:
        cam_topic = hsv_topic
    else:
        cam_topic = right_aruco_topic

    for topic, msg, t in bag.read_messages(topics=[cam_topic]):
        if t < t_frame[1] and t > t_frame[0]:
            total_msgs += 1
            for o in filter(lambda x: x.id != 0, msg.markers):
                obj_dict[o.id] += 1
        elif t < t_frame[0]:
            continue
        elif t > t_frame[1]:
            env_vec = []

            for k in obj_dict.keys():
                if total_msgs != 0:
                    env_vec.append(int(round(obj_dict[k] / total_msgs)))
                else:
                    env_vec.append(0)

            # logging.debug(obj_dict)
            obj_dict = obj_dict.fromkeys(obj_dict, 0)
            total_msgs = 0.
            vec_of_vecs.append(env_vec)

            try:
                t_frame = t_frames.pop(0)
            except IndexError:
                bag.close()
                return vec_of_vecs

    # This bit of repeated code accounts for the case when the calculated duration
    # of the final state is greater than the actual last timestamp of
    # aruco/hsv msggs
    env_vec = []

    for k in sorted(obj_dict.keys()):
        if total_msgs != 0:
            obj_state = round(obj_dict[k] / total_msgs)
            env_vec.append(int(obj_state))
        else:
            env_vec.append(0)

    # logging.debug(obj_dict)
    obj_dict = obj_dict.fromkeys(obj_dict, 0)
    total_msgs = 0.
    vec_of_vecs.append(env_vec)
    bag.close()
    return vec_of_vecs


def match_speech_with_state_vec(bag_path, vocab):
    """Concatenates speech and state vecs into"""
    # Turn OrdereredDicts into lists for easy manipulation
    left_t_frames = get_time_btw_state(left_state_topic, bag_path)
    right_t_frames = get_time_btw_state(right_state_topic, bag_path)

    left_vec = get_env_vec_for_arm(left_state_topic, left_obj_dict, bag_path)
    right_vec = get_env_vec_for_arm(right_state_topic, right_obj_dict,
                                    bag_path)

    state_vecs = []  # an ordered list of state + speech vecs
    labels = []  # corresponding action labels

    # Item 0 of tuple labels the arm for easy parsing below
    # Item 1 is the label
    # item 2 is the time of the state change (i.e. WORKING)
    l_state_changes = [("L", i[1], i[0][1]) for i in left_t_frames[0:-1]]
    r_state_changes = [("R", i[1], i[0][1]) for i in right_t_frames[0:-1]]

    # sort by time
    srted_state_changes = (l_state_changes + r_state_changes)
    srted_state_changes.sort(key=lambda x: x[2])

    l_i = 0
    r_i = 0

    for s in srted_state_changes:
        state_vecs.append(right_vec[r_i] + left_vec[l_i])
        labels.append(s[1])
        if s[0] == "L":
            l_i += 1
        else:
            r_i += 1

    speech_lst = []
    speech_vecs = []
    dict_list = []

    for s in srted_state_changes:
        bag = rosbag.Bag(bag_path)
        min_delta = sys.maxint
        best_s = ""
        for topic, msg, t in bag.read_messages(topics=[speech_topic]):
            # print msg.transcript
            diff = abs(s[2] - t).to_sec()
            if diff < min_delta:
                min_delta = diff
                best_s = msg.transcript
        speech_lst.append(best_s)
        bag.close()
    assert len(speech_lst) == len(state_vecs)
    # Creates a dict with counts for each word that appears in
    # the speech utterance
    logging.debug(speech_lst)
    for i in range(len(speech_lst)):
        v = vocab.fromkeys(vocab, 0)
        sent = speech_lst[i].split(" ")
        for w in sent:
            v[w.lower()] += 1

        state_vecs[i] += v.values()
        speech_vecs.append(v.values())
        dict_list.append(v)

    return state_vecs, labels


def create_vocab(bag_dir_path):
    """Creates a vocabulary used to construct bag of words
    given a dir of rosbags
    """
    vocab = c.OrderedDict()
    for filename in os.listdir(bag_dir_path):
        path = os.path.join(bag_dir_path, filename)
        bag = rosbag.Bag(path)
        for topic, msg, t in bag.read_messages(topics=[speech_topic]):
            sent = msg.transcript.split(" ")
            for w in sent:
                try:
                    vocab[w.lower()] = 0
                except KeyError:
                    pass

        bag.close()
    return vocab


def create_data_set(bag_dir_path, vocab):
    """Given a rosbag dir path or a list of paths, constructs
    Numpy arrays of data and labels to be used"""
    training_X = []
    training_Y = np.array([])
    Xs = []
    Ys = []

    for filename in os.listdir(bag_dir_path):
        path = os.path.join(bag_dir_path, filename)
        X, Y = match_speech_with_state_vec(path, vocab)

        Xs += X
        Ys += Y

    assert len(Xs) == len(Ys)

    for i in range(len(Xs)):
        training_X.append(Xs[i])
        training_Y = np.append(training_Y, Ys[i])
    return np.asarray(training_X), training_Y


def evaluate_model(model, vocab, bag_dir_path):
    X, Y = create_data_set(bag_dir_path, vocab)
    size = len(X)
    rand_indecies = np.random.permutation(size)
    counter = 0

    for test_i in rand_indecies:
        test_X = X[test_i]
        test_Y = Y[test_i]

        train_X = np.delete(X.copy(), (test_i), axis=0)
        train_Y = np.delete(Y.copy(), (test_i), axis=0)

        model.fit(train_X, train_Y)
        prediction = model.predict(test_X.reshape(1, -1))[0]
        actual = test_Y
        if "get_pass_leg" in prediction and "get_pass_leg" in actual:
            counter += 1
        elif prediction == actual:
            counter += 1
        else:
            print("Predicted: {} for (real): {}".format(prediction, actual))

    print("{}% of labels correctly predicited!".format(counter / float(size) * 100))


if __name__ == '__main__':
    v = create_vocab(BAG_DIR_PATH)
    X, Y = create_data_set(BAG_DIR_PATH, v)

    model = GaussianNB()
    model.fit(X, Y)

    with open(MODEL_PATH, "wb") as m:
        joblib.dump(model, m, compress=9)

    with open(VOCAB_PATH, "wb") as m:
        joblib.dump(v, m, compress=9)
