#!/bin/bash
INSTANCES=10

# MAIN ----------------------------------------------------------------------

rm data/*
rm instances/*
python src/data_cleaner.py -i raw_data/clorinda.csv -o instances/ -s $INSTANCES
python src/data_cleaner.py -i raw_data/tartagal.csv -o data/

# Trap ctrl-c and other exit signals and delete all temporary files
trap TrapError 1 2 3 15;
function TrapError() {
    echo "Saliendo...";
    rm data/*
    rm instances/*
    exit;
}
