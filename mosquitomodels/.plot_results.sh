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
    m=$1
    python src/script.py -i data/tartagal.csv --model $m --predict results --sp $2;
}

function plotnonlinear {
    m=$1
    python src/script.py -i data/tartagal.csv -p results/$m.csv --model $m --predict results --sp $2
}

if containsElement "$1" "${nlmodels[@]}"; then
    plotnonlinear $1 $2
elif containsElement "$1" "${lmodels[@]}"; then
    plotlinear $1 $2
fi

# Trap ctrl-c and other exit signals
trap TrapError 1 2 3 15;
function TrapError() {
    echo "Saliendo...";
    exit;
}
