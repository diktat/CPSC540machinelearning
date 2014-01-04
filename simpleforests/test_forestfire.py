import numpy as np
import re

import rf
from tools import *

days = [ 'mon', 'tue',  'wed', 'thu', 'fri', 'sat', 'sun', ]

months = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    ]

def parse_record(record):
    fields = re.split(",", record.strip())

    # the last field is the target
    x, y = fields[:-1], fields[-1]

    # replace string attributes with numeric values
    x[2] = months.index(x[2])
    x[3] = days.index(x[3])

    x = map(float, x)
    y = float(y)

    return x, y

def load_data():
    # http://archive.ics.uci.edu/ml/datasets/Forest+Fires

    with open("data/forestfires.csv") as data_file:
        data = data_file.readlines()

    # The first line is a header
    X, Y = zip(*map(parse_record, data[1:]))

    X = np.asarray(X)
    Y = np.asarray(Y).reshape((-1,1))
    # instructions suggest this is easier to predict
    Y = np.log(1+Y)

    return X, Y

def run_test():
    X, Y = load_data()

    forest = rf.RegressionForest().fit(
            X, Y,
            n_trees=30,
            max_depth=10,
            n_min_leaf=3,
            n_trials=4)
    Yhat = forest.predict(X)

    mse = np.mean((Y - Yhat)**2)
    print mse
    assert mse < 1.0

if __name__ == "__main__":
    run_test()

