#!/bin/bash

# PongNoFrameskip-v4, BreakoutNoFrameskip-v4, NameThisGameNoFrameskip-v4, QbertNoFrameskip-v4, GravitarNoFrameskip-v4, MontezumaRevengeNoFrameskip-v4
ENVS=("NameThisGameNoFrameskip-v4")
SEEDS=(100 200 300) 
ALGO="of4on_direct" # "double_dqn" #"cql"

for env in "${ENVS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        subcmd="sbatch ./sbatch_of4on_small_exp_run.bash $env $seed"
        echo -e "\nSubmit a job to train an of4on policy for the environment, $env, using $ALGO algorithm with the seed $SEED"
        echo -e "Execute the sbatch command:\n\t$subcmd"
        eval $subcmd
    done
done
