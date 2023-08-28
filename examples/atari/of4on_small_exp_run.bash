#!/bin/bash

Task="BreakoutNoFrameskip-v4"

python atari_qrdqn.py --task ${Task} --seed 100 --eps-test 0.001 --eps-train-final 0.01 --buffer-size 1000000 --replay-buffer-min-size 50000 --lr 0.00005 --n-step 1 --target-update-freq 5000 --epoch 6 --step-per-epoch 100000 --step-per-collect 4 --update-per-step 0.25 --training-num 4 --show-progress

python atari_qrdqn.py --task ${Task} --seed 200 --eps-test 0.001 --eps-train-final 0.01 --buffer-size 1000000 --replay-buffer-min-size 50000 --lr 0.00005 --n-step 1 --target-update-freq 5000 --epoch 6 --step-per-epoch 100000 --step-per-collect 4 --update-per-step 0.25 --training-num 4 --show-progress

python atari_qrdqn.py --task ${Task} --seed 300 --eps-test 0.001 --eps-train-final 0.01 --buffer-size 1000000 --replay-buffer-min-size 50000 --lr 0.00005 --n-step 1 --target-update-freq 5000 --epoch 6 --step-per-epoch 100000 --step-per-collect 4 --update-per-step 0.25 --training-num 4 --show-progress

python atari_of4on.py --task ${Task} --seed 100 --offline-epoch-match-consumed-online-steps --online-epoch 6 --show-progress

python atari_of4on.py --task ${Task} --seed 200 --offline-epoch-match-consumed-online-steps --online-epoch 6 --show-progress

python atari_of4on.py --task ${Task} --seed 300 --offline-epoch-match-consumed-online-steps --online-epoch 6 --show-progress
