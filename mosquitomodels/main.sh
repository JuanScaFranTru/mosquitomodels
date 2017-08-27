#!/bin/bash
declare -a nlmodels=($(ls parameters))
declare -a lmodels=('linear' 'ridge')

./clean_data.sh;

if containsElement "$1" "${nlmodels[@]}"; then
    m=$1;
    ./tune_params.sh $m;
    python src/nonlinear.py -i data/tartagal.csv -p results/$m.csv --model $m --predict results

elif containsElement "$1" "${lmodels[@]}"; then
    python src/linear.py  -i data/tartagal.csv --model $m --predict results

else
    for m in "${lmodels[@]}"; do
        python src/nonlinear.py -i data/tartagal.csv -p results/$m.csv --model $m --predict results
    done

    for m in "${nlmodels[@]}"; do
        ./tune_params.sh $m;
        python src/linear.py  -i data/tartagal.csv --model $m --predict results;
    done

fi
