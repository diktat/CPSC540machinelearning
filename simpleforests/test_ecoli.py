import numpy as np
import matplotlib.pyplot as plt
import re

import rf
from tools import *

def parse_record(record):
    fields = re.split("\s+", record.strip())

    # The first field is a name (non-predictive) and the last field is the label
    x, y = fields[1:-1], fields[-1]

    x = map(float, x)

    return x, y

def load_data():
    # http://archive.ics.uci.edu/ml/datasets/Ecoli

    with open("data/ecoli.data") as data_file:
        data = data_file.readlines()

    X, Y = zip(*map(parse_record, data))
    X = np.asarray(X)

    class_names = list(set(Y))
    Y = np.asarray([ class_names.index(y) for y in Y ])

    return X, Y, class_names

def run_test():
    X, Y, class_names = load_data()

    forest = rf.ClassificationForest().fit(
            X,
            encode_one_of_n(Y),
            n_trees=10,
            max_depth=7,
            n_min_leaf=1,
            n_trials=4)
    Yhat = forest.predict(X)
    Yhat = max_of_n_prediction(Yhat)

    acc = np.mean(Y == Yhat)
    print acc
    assert acc > 0.95

if __name__ == "__main__":
    run_test()

