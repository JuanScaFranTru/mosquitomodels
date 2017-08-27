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
    program='library("irace"); load("results/'$model'.Rdata"); a = getFinalElites(iraceResults, n = 0); a$.ID. = NULL; a$model = NULL; a$.PARENT. = NULL; write.csv(a, file = "results/'$model'.csv"); q();'

    echo $program | R --no-save > /dev/null;
}


# MAIN ----------------------------------------------------------------------

if containsElement "$1" "${models[@]}"; then
   model=$1;
   run $model;
elif [ "$1" = "all" ]; then
    for model in "${models[@]}"; do
        run $model;
    done
elif [ "$1" = "clean" ]; then
    rm *.stdout *.stderr -f;
else
    # If the number of arguments is 0 or the argument is incorrect, show help message
    echo -e "${bold}Please, specify a valid model as fisrt argument:";
    echo -e "Models:"
    for model in "${models[@]}"; do
        echo -e "\t $model";
    done
    echo -e "";
    echo -e "${bold}Or specify a valid command as first argument:";
    echo -e "Commands:";
    echo -e "\t all   \t -- Run all models sequentially";
    echo -e "\t clean \t -- Delete all *.stderr and *.stdout files";

    # Exit with error
    exit 1;
fi

# Trap ctrl-c and other exit signals and delete all temporary files
trap TrapError 1 2 3 15;
function TrapError() {
    echo "Saliendo...";
    exit;
}
