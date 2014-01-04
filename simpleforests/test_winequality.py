import numpy as np
import re

import rf
from tools import *

def parse_record(record):
    fields = re.split(";", record.strip())

    # the last field is the target
    x, y = fields[:-1], fields[-1]

    x = map(float, x)
    y = float(y)

    return x, y

def load_data():
    # http://archive.ics.uci.edu/ml/datasets/Ecoli

    with open("data/winequality-red.csv") as data_file:
        data = data_file.readlines()

    # The first line is a header
    X, Y = zip(*map(parse_record, data[1:]))

    X = np.asarray(X)
    Y = np.asarray(Y).reshape((-1,1))

    return X, Y

def run_test():
    X, Y = load_data()

    forest = rf.RegressionForest().fit(
            X,
            Y,
            n_trees=10,
            max_depth=7,
            n_min_leaf=2,
            n_trials=5)
    Yhat = forest.predict(X)

    mse = np.mean((Y - Yhat)**2)
    print mse
    assert mse < 0.25

if __name__ == "__main__":
    run_test()

