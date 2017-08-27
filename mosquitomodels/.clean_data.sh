#!/bin/bash
INSTANCES=5
OVERLAPPING=0.5

# MAIN ----------------------------------------------------------------------

rm data/* -f
rm instances/* -f

python src/data_cleaner.py -i raw_data/clorinda.csv -o instances/ -s $INSTANCES -p $OVERLAPPING
python src/data_cleaner.py -i raw_data/pampa.csv -o instances/ -s $INSTANCES -p $OVERLAPPING
python src/data_cleaner.py -i raw_data/iguazu.csv -o instances/ -s $INSTANCES -p $OVERLAPPING

python src/data_cleaner.py -i raw_data/tartagal.csv -o data/tartagal.csv
python src/data_cleaner.py -i raw_data/clorinda.csv -o data/clorinda.csv
python src/data_cleaner.py -i raw_data/pampa.csv -o data/pampa.csv
python src/data_cleaner.py -i raw_data/iguazu.csv -o data/iguazu.csv

# Trap ctrl-c and other exit signals and delete all temporary files
trap TrapError 1 2 3 15;
function TrapError() {
    echo "Saliendo...";
    rm data/* -f
    rm instances/* -f
    exit;
}
