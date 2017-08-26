#!/usr/bin/python3
"""Train a model
Usage:
  ./train.py -i <filename> -a <r> -n <n> -h <n>
  ./train.py -h | --help
Options:
  -i <filename> Instance filename (data)
  -a <r>        Alpha
  -n <n>        Neurons
  -h <n>        Hidden layers
  -h --help     Show this screen.
"""
from docopt import docopt
from models.utils import load_data, stats
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


def MLP_Regressor(alpha=0.0001, neurons=1, layers=2):
    hidden_layer_sizes = tuple([neurons] * layers)
    return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                        solver='lbfgs', activation='logistic',
                        alpha=alpha)


models = {
    'mlpr': MLP_Regressor,
    'svr': SVR,
    'knnr': KNeighborsRegressor,
    'dtr': DecisionTreeRegressor
}

if __name__ == '__main__':
    opts = docopt(__doc__)

    # Read the penalty parameter
    filename, modelname, opts = parse_docopt()

    data_len, weeks, ts, xs = load_data(filename=filename)

    model = models[modelname]
    model = model(**opts)

    # ------------------- Fitting the model selected ----------------------
    model.fit(xs, ts)

    mean, std_dev = stats(xs, ts, model)
    print(mean)
