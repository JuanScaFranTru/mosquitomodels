"""Run actions using this script.

Usage:
  ./controller.py clear
  ./controller.py [clean] [tune] [plot] [<model> ...]

Options:
  --help                   show this screen
"""
from docopt import docopt
from os import listdir
from os.path import isfile, join
from subprocess import run
from os import remove as remove_file
from glob import glob


PARAMETERS = 'parameters'
LINEARMODELS = ['linear', 'ridge']
ALLMODELS = [f for f in listdir(PARAMETERS) if isfile(join(PARAMETERS, f))]
ALLMODELS += LINEARMODELS


def clean_data():
    print('Cleaning data ...')
    run(['./.clean_data.sh'])


def tune_params(model):
    print('Tuning parameters ...')
    run(['./.tune_params.sh', model])


def plot_results(model):
    run(['./.plot_results.sh', model])


def clear():
    print('Deleting stderr and stdout files ...')
    files = glob('*stderr') + glob('*stdout')
    for f in files:
        remove_file(f)


def print_usage():
    print(__doc__)
    print("Available models:")
    print('{1}{0}'.format('\n  # '.join(ALLMODELS), '  # '))


if __name__ == '__main__':
    opts = docopt(__doc__)

    if opts['clear']:
        clear()
        exit()

    tune = opts['tune']
    clean = opts['clean']
    plot = opts['plot']
    models = list(set(opts['<model>']))  # Unique

    if not (plot or clean or tune or models):
        print_usage()

    if not (set(models) <= set(ALLMODELS)):
        print_usage()
        exit(1)

    if clean:
        clean_data()
    for model in models:
        if tune:
            tune_params(model)
        if plot:
            plot_results(model)
