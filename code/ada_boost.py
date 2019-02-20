from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from prepare_data import get_all_landmarks_emos_labels
import numpy as np
from common_defs import *
from datetime import datetime, timedelta
import time

AB_TRAIN_PERCENT = 0.8

ab_log = "logs/ab.txt"

class Data():
    def __init__(self, X, y):
        self.X = X
        self.y = y


from emos import EMO_STR

def ab_stats(measured, classified):
    f_measure_stats = {}
    for emo in range(1, 8):
        # two of them identified as emo:
        tp = len(np.intersect1d(np.where(classified == emo), np.where(measured == emo)))
        # classified but not measured as emo:
        fp = len(np.intersect1d(np.where(classified == emo), np.where(measured != emo)))
        # not classified but  measured as emo:
        fn = len(np.intersect1d(np.where(classified != emo), np.where(measured == emo)))
        # none of them identified as emo:
        tn = len(np.intersect1d(np.where(classified == emo), np.where(measured == emo)))
        f_measure_stats[EMO_STR[emo-1]] = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
    return f_measure_stats

# X = landmarks
# Y = label in [1, 7]
def set_data_for_ab():
    #for img_normalized_lmark in get_normalized_landmarks_from_video(video_fname):
    X, y = get_all_landmarks_emos_labels()
    return X, y

# Split to train/test
def split_ab_data(X, y):
    rnd_indices = np.random.rand(len(X)) < AB_TRAIN_PERCENT
    train = Data(X[rnd_indices], y[rnd_indices])
    test = Data(X[~rnd_indices], y[~rnd_indices])
    return train, test

def ab_train_and_test(classifier, train, test, desc):
    # Train
    time_start = time.time()
    classifier.fit(train.X, train.y)
    time_end = time.time()
    # Test
    samme_classified = list(classifier.staged_predict(test.X))
    # Accuracy
    accuracy_lst = []
    for test_predict in  classifier.staged_predict(test.X):
        #test_errors.append(1. - accuracy_score(test_predict, test.y))
        accuracy_lst.append(accuracy_score(test_predict, test.y))
    accuracy = np.average(accuracy_lst)
    # Stats & Log
    stats = ab_stats(test.y, np.array(samme_classified))
    # Log
    write_log(ab_log, "AdaBoost: %s, %s" % (desc, datetime.now()))
    write_log(ab_log, "Stats: %s" % str(stats))
    write_log(ab_log, "Accuracy: %s" % accuracy)
    write_log(ab_log, "Eval time: %s" % str(timedelta(seconds=time_end - time_start)))

DEFAULT_MAX_DEPTH = 2
DEFAULT_N_ESTIMATORS = 600
AB_DEFAULT_LEARNING_RATE_SAMME = 1
AB_DEFAULT_LEARNING_RATE_SAMMER = 1

def ab_main(max_dept=DEFAULT_MAX_DEPTH, n_estimators=DEFAULT_N_ESTIMATORS,
            learning_rate_samme=AB_DEFAULT_LEARNING_RATE_SAMME, learning_rate_sammer=AB_DEFAULT_LEARNING_RATE_SAMMER,
            train = None, test = None):
    # Setup the data
    if train is None or test is None:
        X, y = set_data_for_ab()
        train, test = split_ab_data(X, y)

    # SAMME
    bdt_discrete = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=max_dept),
        n_estimators=n_estimators,
        learning_rate=learning_rate_samme,
        algorithm="SAMME")
    ab_train_and_test(bdt_discrete, train, test, "SAMME, max_depth %d, n_estimators=%s, learning_rate=%s" % (max_dept, n_estimators, learning_rate_samme))

    # SAMME.R
    bdt_real = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=max_dept),
        n_estimators=n_estimators,
        learning_rate=learning_rate_sammer)
    ab_train_and_test(bdt_real, train, test, "SAMMER, max_depth %d, n_estimators=%s, learning_rate=%s" % (max_dept, n_estimators, learning_rate_sammer))

def ab_tuning():
    X, y = set_data_for_ab()
    train, test = split_ab_data(X, y)
    for max_depth in [1, 2, 3, 4, 5]:
        for n_estimators in [300, 600, 900, 1200]:
            for rate in [0.1, 0.5, 1, 1.5, 2.0]:
                ab_main(max_depth, n_estimators, rate, rate, train, test)

# def ab_plots():
#     # AdaBoost: SAMME, max_depth 1, n_estimators=300, learning_rate=0.1, 2018-08-09 03:04:47.698042
#     # Stats: {'Anger': {'TP': 187, 'FP': 577, 'TN': 187, 'FN': 206}, 'Contempt': {'TP': 4, 'FP': 53, 'TN': 4, 'FN': 53}, 'Disgust': {'TP': 97, 'FP': 349, 'TN': 97, 'FN': 164}, 'Fear': {'TP': 58, 'FP': 208, 'TN': 58, 'FN': 108}, 'Happy': {'TP': 289, 'FP': 808, 'TN': 289, 'FN': 150}, 'Sadness': {'TP': 28, 'FP': 158, 'TN': 28, 'FN': 117}, 'Surprise': {'TP': 248, 'FP': 685, 'TN': 248, 'FN': 167}}
#     # Accuracy: 0.4945535964175762
#     # Eval time: 0:00:19.971430
#     f = open(ab_log, "r")
#
#     ab_line = f.readline()
#     stat_line = f.readline()
#     acc_line = f.readline()
#     etime_line = f.readline()
#
#     f.close()