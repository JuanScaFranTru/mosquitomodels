"""Train a model

Usage:
  ./script.py -i <s> --model <s> param <n> ...
  ./script.py -i <s> --model <s> --predict <s> --sp <n> -p <s>
  ./script.py -i <s> --model <s> --predict <s> --sp <n>
  ./script.py -h | --help

"""
from utils import load_data, stats, print_stats, save_data, save_plot
from sklearn.decomposition import PCA
from models import MODELS
from pandas import read_csv
import sys


def can_cast(s, caster=int):
    try:
        caster(s)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    argv, opts = sys.argv[1:], {}
    for i in range(0, len(argv) - 1, 2):
        opts[argv[i]] = argv[i+1]

    fixed_opts = ['--predict', '-i', '--sp', '-p', '--model']
    params = {k: v for k, v in opts.items() if k not in fixed_opts}
    opts = {k: v for k, v in opts.items() if k in fixed_opts}

    for k, v in params.items():
        if can_cast(v, int):
            params[k] = int(v)
        elif can_cast(v, float):
            params[k] = float(v)

    model = MODELS[opts['--model']]

    if '-p' in opts:
        params = dict(read_csv(opts['-p']))
        params = {k: v[0] for k, v in params.items()}

    n_cols, weeks, y, X = load_data(filename=opts['-i'])

    try:
        model.pca
    except AttributeError:
        pass
    else:
        pca = PCA(n_components=params['n_components'])
        del params['n_components']
        X = pca.fit(X).transform(X)

    model = model(**params)
    scores, mean, std_dev = stats(X, y, model)

    if '--predict' not in opts:
        print(mean)
    else:
        modelname = opts['--model']
        predict = opts['--predict']

        print_stats(scores, mean, std_dev, 'Stats of ' + modelname)

        results_filename = predict + '/pred-' + modelname + '.csv'
        results_plot_filename = predict + '/pred-' + modelname + '.eps'

        model.fit(X, y)
        y_true, y_pred = y, model.predict(X)
        save_data(results_filename, weeks, y_true, y_pred)
        save_plot(results_plot_filename, weeks, y_true, y_pred)

        results_filename = predict + '/split-pred-' + modelname + '.csv'
        results_plot_filename = predict + '/split-pred-' + modelname + '.eps'

        sp = int(float(opts['--sp']) * len(y))
        model.fit(X[:sp], y[:sp])
        y_true, y_pred = y, model.predict(X)
        save_data(results_filename, weeks, y_true, y_pred)
        save_plot(results_plot_filename, weeks, y_true, y_pred)
