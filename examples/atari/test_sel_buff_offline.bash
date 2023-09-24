#!/bin/bash
Tianshou_Root=/home/jsu/Desktop/Projects/Offline4Online/tianshou
CQL_Script=${Tianshou_Root}/examples/offline/atari_cql.py
Seed=28980811
LR=0.00005
MinQWeight=4.0
Epoch=100
UpdatePerEpoch=12500
BatchSize=32
TestEpisodes=50
TargetUpdateFreq=1000
LogDir=${Tianshou_Root}/examples/atari/log1M_SelExpForOffline
BasicTrainingConfig="--logdir ${LogDir} --lr ${LR} --min-q-weight ${MinQWeight} --epoch ${Epoch} --update-per-epoch ${UpdatePerEpoch} --batch-size ${BatchSize} --test-num ${TestEpisodes} --target-update-freq ${TargetUpdateFreq}"

Task="PongNoFrameskip-v4"
TrainingConfig="${BasicTrainingConfig} --task ${Task} --seed ${Seed}"
echo -e "[Pong: use top 10% episodes in the buffer]"
python ${CQL_Script} ${Training_Config} --top-x-percent 10 --load-buffer-name /home/jsu/Desktop/Projects/Offline4Online/tianshou/examples/atari/log1M_SelExpForOffline/PongNoFrameskip-v4/qrdqn/28980811/230922-145959/buffer_1000000_top10.hdf5

echo -e "[Pong: use top 20% episodes in the buffer]"
python ${CQL_Script} ${Training_Config} --top-x-percent 20 --load-buffer-name /home/jsu/Desktop/Projects/Offline4Online/tianshou/examples/atari/log1M_SelExpForOffline/PongNoFrameskip-v4/qrdqn/28980811/230922-145959/buffer_1000000_top20.hdf5

echo -e "[Pong: use top 50% episodes in the buffer]"
python ${CQL_Script} ${Training_Config} --top-x-percent 50 --load-buffer-name /home/jsu/Desktop/Projects/Offline4Online/tianshou/examples/atari/log1M_SelExpForOffline/PongNoFrameskip-v4/qrdqn/28980811/230922-145959/buffer_1000000_top50.hdf5


Task="NameThisGameNoFrameskip-v4"
TrainingConfig="${BasicTrainingConfig} --task ${Task} --seed ${Seed}"
echo -e "[NameThisGame: extract several subsets from the buffer]"
python /home/jsu/Desktop/Projects/Offline4Online/tianshou/examples/atari/investigate_saved_buffer.py --saved-buffer-filepath /home/jsu/Desktop/Projects/Offline4Online/tianshou/examples/atari/log1M_SelExpForOffline/NameThisGameNoFrameskip-v4/qrdqn/28980811/230922-010231/buffer_1000000.hdf5

echo -e "[NameThisGame: use top 10% episodes in the buffer]"
python ${CQL_Script} ${Training_Config} --top-x-percent 10 --load-buffer-name /home/jsu/Desktop/Projects/Offline4Online/tianshou/examples/atari/log1M_SelExpForOffline/NameThisGameNoFrameskip-v4/qrdqn/28980811/230922-010231/buffer_1000000_top10.hdf5

echo -e "[NameThisGame: use top 20% episodes in the buffer]"
python ${CQL_Script} ${Training_Config} --top-x-percent 20 --load-buffer-name /home/jsu/Desktop/Projects/Offline4Online/tianshou/examples/atari/log1M_SelExpForOffline/NameThisGameNoFrameskip-v4/qrdqn/28980811/230922-010231/buffer_1000000_top20.hdf5

echo -e "[NameThisGame: use top 50% episodes in the buffer]"
python ${CQL_Script} ${Training_Config} --top-x-percent 50 --load-buffer-name /home/jsu/Desktop/Projects/Offline4Online/tianshou/examples/atari/log1M_SelExpForOffline/NameThisGameNoFrameskip-v4/qrdqn/28980811/230922-010231/buffer_1000000_top50.hdf5

