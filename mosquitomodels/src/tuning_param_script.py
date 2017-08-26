"""Train a model

Usage:
  ./tuning_param_script.py -i <s> --model <s> --splitter <s> --min_samples_split <n> --max_depth <n> --min_samples_leaf <n>
  ./tuning_param_script.py -i <s> --model <s> --weights <s> --n_neighbors <n>
  ./tuning_param_script.py -i <s> --model <s> --alpha <n> --neurons <n> --layers <n>
  ./tuning_param_script.py -i <s> --model <s> --C <n> --epsilon <n> --gamma <n> --max_iter <n>
  ./tuning_param_script.py -h | --help

Options:
  -i <s>                   Input file
  --model <s>              Model to be selected

  --splitter <s>           dtr parameter
  --min_samples_split <n>  dtr parameter
  --max_depth <n>          dtr parameter
  --min_samples_leaf <n>   dtr parameter

  --weights <n>            knnr parameter
  --n_neighbors <n>        knnr parameter

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
from utils import load_data, stats
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


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

    del opts['-h']
    del opts['--help']
    del opts['-i']
    del opts['--model']

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

    return filename, modelname, opts


def MLPRegressorProxy(alpha=0.0001, neurons=1, layers=2):
    hidden_layer_sizes = tuple([neurons] * layers)
    return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                        solver='lbfgs', activation='logistic',
                        alpha=alpha)


models = {
    'dtr': DecisionTreeRegressor,
    'knnr': KNeighborsRegressor,
    'mlpr': MLPRegressorProxy,
    'svr': SVR,
}


if __name__ == '__main__':
    filename, modelname, opts = parse_docopt(__doc__)

    print(opts)
    _, _, ts, xs = load_data(filename=filename)

    model = models[modelname]
    model = model(**opts)

    # ------------------- Fitting the model selected ----------------------
    mean, std_dev = stats(xs, ts, model)
    print(mean)
