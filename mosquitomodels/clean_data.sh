#!/bin/bash
INSTANCES=5

# MAIN ----------------------------------------------------------------------

rm data/* -f
rm instances/* -f
python src/data_cleaner.py -i raw_data/clorinda.csv -o instances/ -s $INSTANCES
python src/data_cleaner.py -i raw_data/tartagal.csv -o data/all.csv

# Trap ctrl-c and other exit signals and delete all temporary files
trap TrapError 1 2 3 15;
function TrapError() {
    echo "Saliendo...";
    rm data/* -f
    rm instances/* -f
    exit;
}
