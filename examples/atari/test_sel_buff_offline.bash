#!/bin/bash
TianshouRoot=/home/jsu/Desktop/Projects/Offline4Online/tianshou
Baseline=qrdqn
DatasetExtractScript=${TianshouRoot}/examples/atari/investigate_saved_buffer.py
CQLScript=${TianshouRoot}/examples/offline/atari_cql.py
Seed=28980811
LR=0.0001 # 0.00005, 0.0001, 0.0005
BatchSize=64 # 32, 64, 128
MinQWeight=4.0 # 0.5 (40M), 1.0 (20M), 4.0 (2M), 4/10.0 (0.5M), 4/10/16 (0.2M), 4/10/16/20 (0.1M)
Epoch=50 # 50, 100
UpdatePerEpoch=25000 # 12500, 25000
TestEpisodes=50
TargetUpdateFreq=5000 # 500, 1000, 2000, 5000
LogDir=${TianshouRoot}/examples/atari/log1M_SelExpForOffline
BasicTrainingConfig="--logdir ${LogDir} --lr ${LR} --min-q-weight ${MinQWeight} --epoch ${Epoch} --update-per-epoch ${UpdatePerEpoch} --batch-size ${BatchSize} --test-num ${TestEpisodes} --target-update-freq ${TargetUpdateFreq}"


Game=Pong
Task="${Game}NoFrameskip-v4"
BaselineExpTime="230922-145959"
DatasetFolder="${LogDir}/${Task}/${Baseline}/${Seed}/${BaselineExpTime}"
TrainingConfig="${BasicTrainingConfig} --task ${Task} --seed ${Seed}"
TopXPercents=(10) #(10 20 50)
echo -e "[${Game}: extract several subsets from the buffer]"
cmd="python ${DatasetExtractScript} --saved-buffer-filepath ${DatasetFolder}/buffer_1000000.hdf5"
echo -e "[CMD] ${cmd}"
for top_x_percent in "${TopXPercents[@]}"
do
    echo -e "[${Game}: use top ${top_x_percent}% episodes in the buffer]"
    cmd="python ${CQLScript} ${TrainingConfig} --top-x-percent ${top_x_percent} --load-buffer-name ${DatasetFolder}/buffer_1000000_top${top_x_percent}.hdf5"
    echo -e "[CMD]: ${cmd}"
    eval ${cmd}
done


#Game=NameThisGame
#Task="${Game}NoFrameskip-v4"
#BaselineExpTime="230922-010231"
#DatasetFolder="${LogDir}/${Task}/${Baseline}/${Seed}/${BaselineExpTime}"
#TrainingConfig="${BasicTrainingConfig} --task ${Task} --seed ${Seed}"
#TopXPercents=(10 20 50)
#echo -e "[${Game}: extract several subsets from the buffer]"
#cmd="python ${DatasetExtractScript} --saved-buffer-filepath ${DatasetFolder}/buffer_1000000.hdf5"
#echo -e "[CMD] ${cmd}"
#eval ${cmd}
#for top_x_percent in "${TopXPercents[@]}"
#do
#    echo -e "[${Game}: use top ${top_x_percent}% episodes in the buffer]"
#    cmd="python ${CQLScript} ${TrainingConfig} --top-x-percent ${top_x_percent} --load-buffer-name ${DatasetFolder}/buffer_1000000_top${top_x_percent}.hdf5"
#    echo -e "[CMD]: ${cmd}"
#    eval ${cmd}
#done

