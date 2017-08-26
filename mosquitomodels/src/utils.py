import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score


def to_classes(xs, start=0, stop=1, nclasses=10):
    """Return a Classes array"""
    classified = xs
    intervals = np.linspace(start, stop, nclasses+1)
    for i in range(len(xs)):
        j = 0
        while not (xs[i] >= intervals[j] and xs[i] <= intervals[j + 1]):
            j += 1
            if j == len(intervals) - 1:
                break
        classified[i] = j
    assert len(xs) == len(classified)
    return classified


def smoothing(xs):
    n = len(xs)
    for i in range(1, n - 1):
        xs[i] = sum([xs[j] for j in [i - 1, i, i + 1]]) / 3
    xs[0] = xs[1]
    xs[n - 1] = xs[n - 2]

    return xs


def load_data(cols=None, filename='data/all_data.csv', split=None):
    """Load the training data set.
    Preconditions: 0 <= split <= 1

    split -- percentage of dataset (e.g. 0.9). If None, no splitting is done.
    """

    assert split is None or 0 <= split <= 1

    if cols is None:
        cols = tuple(range(2, 34))

    weeks = np.loadtxt(filename, delimiter=',', usecols=0, dtype=float)
    ts = np.loadtxt(filename, delimiter=',', usecols=1, dtype=float)
    xs = np.loadtxt(filename, delimiter=',', usecols=cols, dtype=float)
    data_len = len(weeks)

    ts = smoothing(ts)

    if split is None:
        return data_len, weeks, ts, xs
    else:
        # Reserve split * 100% of the data for validation
        split_point = int(data_len * split)
        ts, vts = ts[0:split_point], ts[split_point:data_len]
        xs, vxs = xs[0:split_point, :], xs[split_point:data_len, :]
        weeks, vweeks = weeks[0:split_point], weeks[split_point:data_len]
        return data_len, weeks, ts, xs, vweeks, vts, vxs


def stats(xs, ts, model, n_splits=5):
    cv = TimeSeriesSplit(n_splits)
    scores = cross_val_score(model, xs, ts, cv=cv.split(xs),
                             scoring='neg_mean_squared_error')
    scores = np.sqrt(-scores)
    mean = np.mean(scores)
    std_dev = np.std(scores)

    return scores, mean, std_dev


def print_stats(xs, ts, model, title='Stats', n_splits=5):
    scores, mean, std_dev = stats(xs, ts, model, n_splits)

    print()
    print(title)
    print('-' * len(title))
    print('')
    print('Model Scores: ', scores)
    print('Mean Score: ', mean)
    print('Standard Deviation of Score: ', std_dev)


def plot_prediction(weeks, xs, ts, model):
    ys_pred = model.predict(xs)
    plt.plot(weeks, ts)
    plt.plot(weeks, ys_pred)
    plt.show()


def plot(xs, ys):
    plt.plot(xs, ys)

def show():
    plt.show()
