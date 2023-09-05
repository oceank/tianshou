#!/bin/sh
#SBATCH --job-name=sm_oo
#SBATCH -N 1
#SBATCH -n 32   ## how many cpu cores to request
#SBATCH --gres=gpu:1   ## Run on 1 GPU
#SBATCH --output /work/suj/of4on/slurm/sm_oo_%j.out
#SBATCH --error //work/suj/of4on/slurm/sm_oo_%j.err
#SBATCH -p dgx_aic
# AI_Center,gpu-v100-16gb,gpu-v100-32gb,v100-16gb-hiprio,v100-32gb-hiprio,

#SBATCH --exclude=node[363,370]

#export SLURM_MEM_BIND="local"
#export SLURM_CPU_BIND="map_cpu:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
#export SLURM_CPU_BIND="map_cpu:16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"

# Accessibale GPU queues
#   AI_Center,dgx_aic,gpu-v100-16gb,gpu-v100-32gb,v100-16gb-hiprio,v100-32gb-hiprio
#   AI_Center,v100-32gb-hiprio,gpu-v100-32gb
#   gpu-v100-16gb,v100-16gb-hiprio



echo "\n======= Setup the environment =======\n"
module load python3/anaconda/2021.11
module load cuda/11.0
source activate of4on_ts

echo "\n======= Check the environment =======\n"
echo $CONDA_PREFIX
echo $(uname -a)
nvcc --version
nvidia-smi



echo "\n======= Run the task... =======\n"

# PongNoFrameskip-v4, BreakoutNoFrameskip-v4, NameThisGameNoFrameskip-v4, QbertNoFrameskip-v4, GravitarNoFrameskip-v4
Task=$1
Seed=$2

Tianshou_Root=/work/suj/of4on/tianshou
SCRIPT_BASELINE=${Tianshou_Root}/examples/atari/atari_qrdqn.py
SCRIPT_OF4ON=${Tianshou_Root}/examples/atari/atari_of4on.py

Torch_Num_Threads=16 # 2, 4, 8, 16
Train_Env_Num_Threads=4
Test_Env_Num_Threads=5 # 5, 10

#echo -e "Run baseline with seed ${Seed} in the game ${Task}"
#python ${SCRIPT_BASELINE} --task ${Task} --seed ${Seed} --eps-test 0.001 --eps-train-final 0.01 --buffer-size 1000000 --replay-buffer-min-size 50000 --lr 0.00005 --n-step 1 --target-update-freq 5000 --epoch 6 --step-per-epoch 100000 --step-per-collect 4 --update-per-step 0.25 --training-num 4 --torch-num-threads ${Torch_Num_Threads} --train-env-num-threads ${Train_Env_Num_Threads} --test-env-num-threads ${Test_Env_Num_Threads}

echo -e "Run of4on with seed ${Seed} in the game ${Task}"
#python ${SCRIPT_OF4ON} --task ${Task} --seed ${Seed} --offline-epoch-match-consumed-online-steps --online-epoch 6 --torch-num-threads ${Torch_Num_Threads} --train-env-num-threads ${Train_Env_Num_Threads} --test-env-num-threads ${Test_Env_Num_Threads}
#--reset-replay-buffer-per-phase --random-exploration-before-each-phase

# Experience Collection Types: online policy collecting ratio
#   Fixed: 0.75/0.5/0.25
#   Increase: 0.25 -> 0.5
#   Decrease: 0.5 -> 0.25

python ${Tianshou_Root}/examples/atari/atari_of4on_2pc.py --task ${Task} --seed ${Seed} --offline-epoch-match-consumed-online-steps --online-epoch 6 --torch-num-threads ${Torch_Num_Threads} --train-env-num-threads ${Train_Env_Num_Threads} --test-env-num-threads ${Test_Env_Num_Threads} --online-policy-collecting-ratio 0.5 --online-policy-collecting-ratio_final 0.5

