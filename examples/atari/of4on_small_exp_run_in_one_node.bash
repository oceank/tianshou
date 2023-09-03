#!/bin/bash

Task=$1
Tianshou_Root=/work/suj/of4on/tianshou
SCRIPT_BASELINE=${Tianshou_Root}/examples/atari/atari_qrdqn.py
SCRIPT_OF4ON=${Tianshou_Root}/examples/atari/atari_of4on.py
SEEDS=(100 200 300)

Torch_Num_Threads=16 # 2, 4, 8, 16
Train_Env_Num_Threads=4
Test_Env_Num_Threads=5 # 5, 10

for seed in "${SEEDS[@]}"
do
    echo -e "Run baseline with seed ${seed} in the game ${Task}"
    python ${SCRIPT_BASELINE} --task ${Task} --seed ${seed} --eps-test 0.001 --eps-train-final 0.01 --buffer-size 1000000 --replay-buffer-min-size 50000 --lr 0.00005 --n-step 1 --target-update-freq 5000 --epoch 6 --step-per-epoch 100000 --step-per-collect 4 --update-per-step 0.25 --training-num 4 --torch-num-threads ${Torch_Num_Threads} --train-env-num-threads ${Train_Env_Num_Threads} --test-env-num-threads ${Test_Env_Num_Threads}

    #echo -e "Run of4on with seed ${seed} in the game ${Task}"
    #python ${SCRIPT_OF4ON} --task ${Task} --seed ${seed} --offline-epoch-match-consumed-online-steps --online-epoch 6 --torch-num-threads ${Torch_Num_Threads} --train-env-num-threads ${Train_Env_Num_Threads} --test-env-num-threads ${Test_Env_Num_Threads}

done 
