#!/bin/bash
declare -a models=($(ls parameters))

containsElement () {
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}

function run {
    # Please do NOT code like this
    model=$1
    irace -s ./.scenario -p parameters/$model -l results/$model.Rdata;

    # Output results in a csv
    program='library("irace"); load("results/'$model'.Rdata"); a = getFinalElites(iraceResults, n = 0); a$.ID. = NULL; a$model = NULL; a$.PARENT. = NULL; write.csv(a, file = "results/'$model'.csv", row.names=F); q();'

    echo $program | R --no-save > /dev/null;
}


# MAIN ----------------------------------------------------------------------

if containsElement "$1" "${models[@]}"; then
   model=$1;
   run $model;
else
    # If the number of arguments is 0 or the argument is incorrect, show error message
    echo -e "No model was provided";
    # Exit with error
    exit 1;
fi

# Trap ctrl-c and other exit signals and delete all temporary files
trap TrapError 1 2 3 15;
function TrapError() {
    echo "Saliendo...";
    exit;
}
