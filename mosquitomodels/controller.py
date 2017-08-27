"""Run actions using this script.

Usage:
  ./controller.py clear
  ./controller.py [clean] [tune] [plot --sp <n>] [<model> ...]

Options:
  --sp <n>                 splitting point
  --help                   show this screen

Examples:
  python controller.py clear                     # Delete the all *stderr and *stdout
  python controller.py clean                     # Clean the data
  python controller.py tune                      # Use the cleaned data to tune all models
  python controller.py tune svr                  # Tune svr params using the cleaned data
  python controller.py tune svr mlpr             # Tune svr and mlpr params using the cleaned data
  python controller.py plot svr                  # Plot predictions using svr model using the selected parameters
  python controller.py clean tune plot svr mlpr  # Clean, tune and plot svr and mlpr models
  python controller.py clean tune plot linear    # Clean and plot linear model (linear cannot be tuned)
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


def plot_results(model, sp):
    run(['./.plot_results.sh', model, sp])


def clear():
    print('Deleting stderr and stdout files ...')
    files = glob('*stderr') + glob('*stdout')
    for f in files:
        remove_file(f)


if __name__ == '__main__':
    __doc__ += '\nAvailable models:\n'
    __doc__ += '{1}{0}'.format('\n  # '.join(ALLMODELS), '  # ')
    opts = docopt(__doc__)

    if opts['clear']:
        clear()
        exit()

    tune = opts['tune']
    clean = opts['clean']
    plot = opts['plot']
    sp = opts['--sp']
    models = list(set(opts['<model>']))  # Unique

    if not (plot or clean or tune or models):
        print(__doc__)

    if not (set(models) <= set(ALLMODELS)):
        print(__doc__)
        exit(1)

    if clean:
        clean_data()
    for model in models:
        if tune and model not in LINEARMODELS:
            tune_params(model)
        if plot:
            plot_results(model, sp)
