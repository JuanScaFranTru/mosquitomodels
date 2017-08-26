"""Train a model

Usage:
  ./linear.py -i <file> --model <model> [--n_comp <n>] --predict <file>
  ./linear.py -h | --help

Options:
  -i <file>                Input file
  --predict <file>         Predict and save results in the given directory
  --model <model>          Model to be selected

  --n_comp <n>             PCA_model parameter

  --help                   show this screen
"""
from docopt import docopt
from utils import load_data, stats, print_stats, save_data, save_plot
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
import numpy as np


class PositiveRidgeCV(RidgeCV):
    def predict(self, X):
        y = super(PositiveRidgeCV, self).predict(X)
        y = np.maximum([0] * len(y), y)
        return y


class PositiveLinearRegression(LinearRegression):
    def predict(self, X):
        y = super(LinearRegression, self).predict(X)
        y = np.maximum([0] * len(y), y)
        return y


models = {
    'linear': PositiveLinearRegression,
    'ridge': PositiveRidgeCV,
}


if __name__ == '__main__':
    opts = docopt(__doc__)
    filename = opts['-i']
    predict = opts['--predict']

    modelname = opts['-m']
    linear_model = models[modelname]()

    data_len, weeks, ts, xs = load_data(filename=filename)

    # Models uses PCA?
    if opts['--n_comp'] is not None:
        n_components = int(opts['--n_comp'])
        pca = PCA(n_components=n_components)
        xs = pca.fit(xs).transform(xs)

    scores, mean, std_dev = stats(xs, ts, linear_model)
    if predict is None:
        print(mean)
    else:
        print_stats(scores, mean, std_dev)

        results_filename = predict + '/prediction-' + modelname + '.csv'
        linear_model.fit(xs, ts)
        ts_pred = linear_model.predict(xs)
        save_data(results_filename, weeks, ts, ts_pred)

        results_plot_filename = predict + '/prediction-' + modelname + '.eps'
        save_plot(results_filename, weeks, ts, ts_pred)
