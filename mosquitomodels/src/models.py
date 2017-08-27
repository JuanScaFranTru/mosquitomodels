"""Train a model

Usage:
  ./models.py -i <s> --model <s> --splitter <s> --min_samples_split <n> --max_depth <n> --min_samples_leaf <n>
  ./models.py -i <s> --model <s> --weights <s> --n_neighbors <n> --algorithm <s> --leaf_size <n>
  ./models.py -i <s> --model <s> --alpha <n> --neurons <n> --layers <n>
  ./models.py -i <s> --model <s> --C <n> --epsilon <n> --gamma <n> --max_iter <n>
  ./models.py -i <s> -p <s> --model <s> --predict  <s> --sp <n>
  ./models.py -i <s> --model <s> --predict <s> --sp <n>
  ./models.py -h | --help

Options:
  -i <s>                   Input file
  -p <s>                   Parameters file
  --sp <n>                 Spliting point
  --model <s>              Model to be selected
  --predict <s>            Predict and save results in the given directory

  --splitter <s>           dtr parameter
  --min_samples_split <n>  dtr parameter
  --max_depth <n>          dtr parameter
  --min_samples_leaf <n>   dtr parameter

  --weights <n>            knnr parameter
  --n_neighbors <n>        knnr parameter
  --algorithm <s>          knnr parameter
  --leaf_size <n>          knnr parameter

  --alpha <n>              mlpr parameter
  --neurons <n>            mlpr parameter
  --layers <n>             mlpr parameter

  --C <n>                  svr parameter
  --epsilon <n>            svr parameter
  --gamma <n>              svr parameter
  --max_iter <n>           svr parameter

  --help                   show this screen
"""
from docopt import docopt
from utils import load_data, stats, print_stats, save_data, save_plot
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from pandas import read_csv


def can_cast(s, caster=int):
    try:
        caster(s)
        return True
    except ValueError:
        return False


def parse_docopt(doc):
    opts = docopt(doc)

    filename = opts['-i']
    modelname = opts['--model']
    predict = opts['--predict']
    paramfilename = opts['-p']
    sp = None
    if predict:
        sp = float(opts['--sp'])

    del opts['-h']
    del opts['--help']
    del opts['-i']
    del opts['--model']
    del opts['--predict']
    del opts['-p']
    del opts['--sp']

    copy = opts.copy()
    for k, v in copy.items():
        del opts[k]
        if v is not None:
            k = k[2:]
            opts[k] = v

    for k, v in opts.items():
        if can_cast(v, int):
            opts[k] = int(v)
        elif can_cast(v, float):
            opts[k] = float(v)

    return paramfilename, filename, modelname, predict, sp, opts


def MLPRegressorProxy(alpha, neurons, layers):
    hidden_layer_sizes = tuple([neurons] * layers)
    return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                        solver='lbfgs', activation='logistic',
                        alpha=alpha)


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
    'dtr': DecisionTreeRegressor,
    'knnr': KNeighborsRegressor,
    'mlpr': MLPRegressorProxy,
    'svr': SVR,
    'linear': PositiveLinearRegression,
    'ridge': PositiveRidgeCV,
}


if __name__ == '__main__':
    paramfilename, filename, modelname, predict, sp, opts = parse_docopt(__doc__)
    model = models[modelname]

    if paramfilename is not None:
        opts = dict(read_csv(paramfilename))
        opts = {k: v[0] for k, v in opts.items()}

    assert opts is not None

    model = model(**opts)

    n_cols, weeks, ts, xs = load_data(filename=filename)
    scores, mean, std_dev = stats(xs, ts, model)

    if predict is None:
        print(mean)
    else:
        print_stats(scores, mean, std_dev, 'Stats of ' + modelname)

        results_filename = predict + '/prediction-' + modelname + '.csv'
        results_plot_filename = predict + '/prediction-' + modelname + '.eps'

        model.fit(xs, ts)
        ts_pred = model.predict(xs)
        save_data(results_filename, weeks, ts, ts_pred)
        save_plot(results_plot_filename, weeks, ts, ts_pred)

        results_filename = predict + '/splited-prediction-' + modelname + '.csv'
        results_plot_filename = predict + '/splited-prediction-' + modelname + '.eps'

        sp = int(sp * len(ts))
        model.fit(xs[:sp], ts[:sp])
        ts_pred = model.predict(xs)
        save_data(results_filename, weeks, ts, ts_pred)
        save_plot(results_plot_filename, weeks, ts, ts_pred)
