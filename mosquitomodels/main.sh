#!/bin/bash
declare -a nlmodels=($(ls parameters))
declare -a lmodels=('linear' 'ridge')

containsElement () {
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}

./clean_data.sh;

if containsElement "$1" "${nlmodels[@]}"; then
    m=$1;
    ./tune_params.sh $m;
    python src/nonlinear.py -i data/tartagal.csv -p results/$m.csv --model $m --predict results

elif containsElement "$1" "${lmodels[@]}"; then
    python src/linear.py  -i data/tartagal.csv --model $m --predict results

else
    for m in "${nlmodels[@]}"; do
        ./tune_params.sh $m;
        python src/nonlinear.py -i data/tartagal.csv -p results/$m.csv --model $m --predict results
    done

    for m in "${lmodels[@]}"; do
        python src/linear.py  -i data/tartagal.csv --model $m --predict results;
    done

fi

# Trap ctrl-c and other exit signals and delete all temporary files
trap TrapError 1 2 3 15;
function TrapError() {
    echo "Saliendo...";
    exit;
}
