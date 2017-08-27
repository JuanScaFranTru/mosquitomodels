# Models

## Install and uninstall
### Install

You must have anaconda installed in order to install the package and the environment.
Download anaconda from [here](https://www.continuum.io/downloads).

```
conda env create -f $ENV;
source activate $ENV_NAME;
```

**Note**: This will create a new environment.

### Uninstall
To uninstall run:

```
source deactivate $ENV_NAME;
conda remove --name $ENV_NAME --all;
```

## Running irace

```
./run_model.sh MODEL
```

to see which models are available:

```
./run_model.sh
```

to run all models:

```
./run_model.sh ALL
```

## Displaying errors

If something went wrong while running irace you can run `cat *err | less` and
`cat *out | less`. The last command shows any output of your script on
`/dev/stdout` and the former does the same but with `/dev/stderr`.


## Plotting input data heatmap

Run:

```
python src/plotdata.py raw_data/clorinda.csv raw_data/iguazu.csv raw_data/pampa.csv raw_data/tartagal.csv 
```
