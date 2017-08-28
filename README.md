# Models

## Install and uninstall
### Installing R in ubuntu 16

```
sudo apt-get install r-base --yes
```


### Installing irace

```
sudo R CMD INSTALL irace
```

Add the following to .zshrc or .bashrc
```
# Irace path
# Path given by system.file(package="irace")
export IRACE_HOME=/usr/local/lib/R/site-library/irace
export PATH=${IRACE_HOME}/bin/:$PATH
```


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

To plot the raw data:

```
python --skipcols 3 --nolag src/plotdata.py raw_data/clorinda.csv raw_data/iguazu.csv raw_data/pampa.csv raw_data/tartagal.csv 
```

To plot the cleaned data with the selected variables:

```
./clean_data.sh
python src/plotdata.py --skipcols 1 data/clorinda.csv data/iguazu.csv data/pampa.csv data/tartagal.csv 
```
