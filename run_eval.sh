dataset=$1 # dataset name (sumtime, sportsett, mlb, obituary)
type=$2 # acc/auto
gpu=$3


if [ $type == "auto" ]; then

    echo "dataset: $dataset"

    echo " "
    echo "Calculating automated metrics"
    echo " "

    mets="bleu,rouge,chrf++,bert,meteor"

    if [ $dataset = "mlb" ]; then
        systems="neural_ent neural_mp"

    elif [ $dataset = "obituary" ]; then
        systems="neural_peg neural_t5 neural_bart"

    elif [ $dataset = "sportsett" ]; then
        systems="neural_ent neural_mp neural_hir"

    elif [ $dataset = "sumtime" ]; then
        systems="neural_peg neural_t5 neural_bart"

    else
        echo "Wrong dataset name"
        exit

    fi

    echo "systems: $systems"
    echo "metrics: $mets"

    for system in $systems
    do
        echo ""
        echo "Evaluating $system for $dataset using automated metrics"
        echo ""
        CUDA_VISIBLE_DEVICES=$gpu python3 eval/auto_evals.py \
                -reference ./$dataset/eval/sys_gens/gold.txt \
                -hypothesis ./$dataset/eval/sys_gens/$system.txt \
                -num_refs 1 \
                -metrics $mets
    done

fi


if [ $type == "acc" ]; then

    echo " "
    echo "Calculating accuracy error score"
    echo " "

    if [ $dataset = "mlb" ]; then
        systems="neural_ent neural_mp"
    elif [ $dataset = "obituary" ]; then
        systems="neural_bart neural_t5 neural_peg"
    elif [ $dataset = "sportsett" ]; then
        systems="neural_mp neural_hir neural_ent"
    elif [ $dataset = "sumtime" ]; then
        systems="neural_bart neural_t5 neural_peg"
    else
        echo "Wrong dataset name"
        exit
    fi

    echo "systems: $systems"

    for system in $systems
    do
        echo ""
        echo "Evaluating $system for $dataset using manual annotations"
        echo ""
        python3 eval/acc_eval.py -dataset $dataset -system $system
    done

fi
