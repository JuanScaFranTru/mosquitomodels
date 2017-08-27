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
    ./tune_params.sh $1;
    ./plot_results.sh $1;
elif containsElement "$1" "${lmodels[@]}"; then
    ./plot_results.sh $1;
else
    for m in "${nlmodels[@]}"; do
        ./tune_params.sh $m;
        ./plot_results.sh $m;
        sleep 100;
    done

    for m in "${lmodels[@]}"; do
        ./plot_results.sh $m;
    done
fi

# Trap ctrl-c and other exit signals and delete all temporary files
trap TrapError 1 2 3 15;
function TrapError() {
    echo "Saliendo...";
    exit;
}
