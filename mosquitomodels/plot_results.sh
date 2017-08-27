#!/bin/bash
declare -a nlmodels=($(ls parameters))
declare -a lmodels=('linear' 'ridge')

containsElement () {
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}

function plotlinear {
    model=$1
    python src/linear.py  -i data/tartagal.csv --model $m --predict results;
}

function plotnonlinear {
    model=$1
    python src/nonlinear.py -i data/tartagal.csv -p results/$m.csv --model $m --predict results
}

if containsElement "$1" "${nlmodels[@]}"; then
    plotnonlinear $1
elif containsElement "$1" "${lmodels[@]}"; then
    plotlinear $1
else
    for m in "${nlmodels[@]}"; do
        plotnonlinear $1
    done

    for m in "${lmodels[@]}"; do
        plotlinear $1
    done
fi

# Trap ctrl-c and other exit signals and delete all temporary files
trap TrapError 1 2 3 15;
function TrapError() {
    echo "Saliendo...";
    exit;
}
