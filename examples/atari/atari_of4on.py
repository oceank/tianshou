import argparse
import datetime
import os
import sys
import pprint
import datetime
import random
from functools import partial
from copy import deepcopy
import json

import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch
from atari_network import QRDQN
from atari_wrapper import make_atari_env, make_atari_env_for_testing_using_envpool
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Of4OnCollector, Collector, VectorReplayBuffer
from tianshou.policy import QRDQNPolicy, DiscreteCQLPolicy
from tianshou.trainer import offpolicy_trainer, offline_trainer, offpolicy_dualpolicycollection_trainer, OnlinePolicyExperienceCollectionSetting
from tianshou.utils import TensorboardLogger, WandbLogger
#from tianshou.utils.net.common import DataParallelNet

from examples.atari.utils import set_torch_seed, set_determenistic_mode

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--train-env-num-threads", type=int, default=4)
    parser.add_argument("--test-env-num-threads", type=int, default=5)
    parser.add_argument("--task", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # state representation and processing
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--scale-obs", type=int, default=0)
    # policy evaluation
    parser.add_argument("--test-num", type=int, default=10) # number of espidoes used to evaluate a policy
    parser.add_argument("--eps-test", type=float, default=0.001)
    # online RL  : QRDQN
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.01)
    parser.add_argument("--exploration-duration", type=int, default=1000000)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--replay-buffer-min-size", type=int, default=50000)
    parser.add_argument("--online_lr", type=float, default=0.00005)
    parser.add_argument("--online-gamma", type=float, default=0.99)
    parser.add_argument("--online-num-quantiles", type=int, default=200)
    parser.add_argument("--online-n-step", type=int, default=1)
    parser.add_argument("--online-target-update-freq", type=int, default=5000)
    parser.add_argument("--online-batch-size", type=int, default=32)
    parser.add_argument("--online-epoch", type=int, default=50)
    parser.add_argument("--online-step-per-epoch", type=int, default=100000)
    parser.add_argument("--online-step-per-collect", type=int, default=4)
    parser.add_argument("--online-update-per-step", type=float, default=0.25)
    parser.add_argument("--online-training-num", type=int, default=4)
    # offlline RL: CQL (QRDQN)
    parser.add_argument("--offline-lr", type=float, default=0.00005)
    parser.add_argument("--offline-gamma", type=float, default=0.99)
    parser.add_argument("--offline-num-quantiles", type=int, default=200)
    parser.add_argument("--offline-n-step", type=int, default=1)
    parser.add_argument("--offline-target-update-freq", type=int, default=1000)
    parser.add_argument("--offline-epoch", type=int, default=25)
    parser.add_argument("--offline-update-per-epoch", type=int, default=50000)
    parser.add_argument("--offline-batch-size", type=int, default=32)
    parser.add_argument("--min-q-weight", type=float, default=4.0)
    # offline for online
    parser.add_argument("--num-phases", type=int, default=5) # by default, one phase has 10 online epochs
    parser.add_argument("--of4on-type", type=str, default="DirectCopy") # DirectCopy, DualCollection, PEX(ToDo)
    ## For DirectCopy
    parser.add_argument("--reset-replay-buffer-per-phase", action="store_true")
    parser.add_argument("--random-exploration-before-each-phase", action="store_true") # for phases with a id > 1
    ## For DualCollect
    parser.add_argument("--online-policy-collecting-setting-type", type=str, default="Fixed") # Fixed, Scheduling, Adaptive
    parser.add_argument("--online-policy-collecting-ratio", type=float, default=1.0) # 1.0, 0.75, 0.5, 0.25
    parser.add_argument("--online-policy-collecting-ratio-start", type=float, default=0.0) # 0.0 , 0.25
    parser.add_argument("--online-policy-collecting-ratio-final", type=float, default=1.0) # 1.0,  0.75
    ## For all of4on types
    parser.add_argument("--transfer-best-offline-policy", action="store_true")
    ## 0: args.offline_epoch, 1/2: 5X of gradients of online policy in the current phase/inferred by the current buffer
    parser.add_argument("--offline-epoch-setting", type=int, default = 0)
    parser.add_argument("--bootstrap-offline-with-online", action="store_true") # copy the best online policy to initialize offline policy
    # other training configuration
    parser.add_argument("--early-stop", type=bool, default=False)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--use-multi-gpus", action="store_true")
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument("--render", type=float, default=0.)
   
    # logging
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )

    return parser.parse_args()

def test_of4on(args=get_args()):
    torch.set_num_threads(args.torch_num_threads)
    
    print(f"[{datetime.datetime.now()}] experiment configuration:", flush=True)
    pprint.pprint(vars(args), indent=4)
    sys.stdout.flush()

    def add_algo_name():
        # The name format (without the seed and the experiment starting time) of log folder for each experiment of the same game:
        # cql-qrdqn-<of4on type>-<of4on type setting>-<'bf' or 'rf'>-ofe-<'phase5grad', 'buffer5grad', 'userdefined'>
        #   <of4on type>: 'dic' (DirectCopy) | 'duc' (DualCollect)
        #   <of4on type setting>: 'f2o' (OfflineToOnline) | ('f'<decimal> (Fixed), 'sb'<ratio start>'e'<ratio end> (Scheduling), 'adp' (Adaptive))
        #   <'bf' or 'rf'>: 'bf' (Transfer BestOfflinePolicy), 'rf' (Transfer RecentOfflinePolicy)
        #   ofe: offline epoch
        #   <'phase5grad', 'buffer5grad', 'userdefined'>: 5X of OnlinePolicyGradientSteps of Current Phase or associated with Curreht Buffer, or a user-defined number
        ## offline and online algorithms' name
        args.algo_name = "cql-qrdqn"
        ## of4on type and type setting
        if args.of4on_type == "DirectCopy":
            args.algo_name += "-dic-f2o"
        elif args.of4on_type == "DualCollect":
            args.algo_name += "-duc"
            if args.online_policy_collecting_setting_type == "Fixed":
                args.algo_name += f"-f{int(100*args.online_policy_collecting_ratio)}"
            elif args.online_policy_collecting_setting_type == "Scheduling":
                args.algo_name += f"-sb{int(100*args.online_policy_collecting_ratio_start)}e{int(100*args.online_policy_collecting_ratio_final)}"
            elif args.online_policy_collecting_setting_type == "Adaptive":
                args.algo_name += "-adp"
            else:
                raise f"Unsupported online policy collecting setting of the DualCollect of4on type: {args.online_policy_collecting_setting_type}"
        else:
            raise f"Unsupported of4on type: {args.of4on_type}"
        ## transfer the best or the recent offline policy to the next phase
        if args.transfer_best_offline_policy:
            args.algo_name += "-bf"
        else:
            args.algo_name += "-rf"
        ## offline epoch setting
        args.algo_name += "-ofe"
        if args.offline_epoch_setting == 1:
            args.algo_name += "-phase5grad"
        elif args.offline_epoch_setting == 2:
            args.algo_name += "-buffer5grad"
        elif args.offline_epoch_setting == 0:
            args.algo_name += "-userdefined"
        else:
            raise f"Unsupported offline epoch setting, {args.offline_epoch_setting}"
    
    add_algo_name()
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name_prefix = os.path.join(args.task, args.algo_name, str(args.seed), now)
    experiment_result_dir = os.path.join(args.logdir, log_name_prefix)
    os.makedirs(experiment_result_dir, exist_ok=True)
    print(f"[{datetime.datetime.now()}] The available number of GPUs: {torch.cuda.device_count()}", flush=True)
    with open(os.path.join(experiment_result_dir, "traning_configuration.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # seed
    disable_cudnn = False
    set_determenistic_mode(args.seed, disable_cudnn)

    # create envs
    env, train_envs, test_envs = make_atari_env(
        args.task,
        args.seed,
        args.online_training_num,
        args.test_num,
        train_env_num_threads = args.train_env_num_threads,
        test_env_num_threads = args.test_env_num_threads,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    env.action_space.seed(args.seed)
    train_envs.action_space.seed(args.seed)
    test_envs.action_space.seed(args.seed)

    # testing envs for offline learning
    # The reason for creating two testing envs is because the instantiation of Collector
    # will reset the input envs, which will cause the testing envs to be reset twice and
    # become different from that in the experiment of baseline (qrdqn).
    test_envs_of = make_atari_env_for_testing_using_envpool(
        args.task, args.seed, args.test_num, args.frames_stack,
        num_threads=args.test_env_num_threads)
    test_envs_of.action_space.seed(args.seed)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print(f"[{datetime.datetime.now()}] Observations shape: {args.state_shape}", flush=True)
    print(f"[{datetime.datetime.now()}] Actions shape: {args.action_shape}", flush=True)

   
    # initialize qrdqn and cql (qrdqn)
    def create_offline_policy(current_best_online_policy_path=""):
        set_torch_seed(args.seed)
        offline_model = QRDQN(*args.state_shape, args.action_shape, args.offline_num_quantiles, args.device)
        offline_optim = torch.optim.Adam(offline_model.parameters(), lr=args.offline_lr, eps=0.01/32)
        offline_policy = DiscreteCQLPolicy(
            offline_model,
            offline_optim,
            args.offline_gamma,
            args.offline_num_quantiles,
            args.offline_n_step,
            args.offline_target_update_freq,
            min_q_weight=args.min_q_weight,
        ).to(args.device)

        if args.bootstrap_offline_with_online:
            # bootstrap offline policy with online policy learned in the current phase
            if current_best_online_policy_path != "":
                offline_policy.load_state_dict(
                    torch.load(current_best_online_policy_path, map_location=args.device)
                )
                offline_policy.sync_weight() # sync the target network with the online network

        return offline_policy

    def create_online_policy(previous_phase_best_offline_policy_path=""):
        set_torch_seed(args.seed)
        online_model = QRDQN(*args.state_shape, args.action_shape, args.online_num_quantiles, args.device)
        online_optim = torch.optim.Adam(online_model.parameters(), lr=args.online_lr, eps=0.01/32)
        online_policy = QRDQNPolicy(
            online_model,
            online_optim,
            args.online_gamma,
            args.online_num_quantiles,
            args.online_n_step,
            target_update_freq=args.online_target_update_freq,
        ).to(args.device)

        if previous_phase_best_offline_policy_path != "":
            # bootstrap online policy with offline policy learned in the previous phase
            online_policy.load_state_dict(
                torch.load(previous_phase_best_offline_policy_path, map_location=args.device)
            )
            online_policy.sync_weight()
        return online_policy

    online_policy = create_online_policy()
    offline_policy = create_offline_policy()

    # load previously trained policies
    if args.resume_path: # resume_path is the path to the directory that stores the online and offline models
        online_policy_path = os.path.join(args.resume_path, "online_policy.pth")
        offline_policy_path = os.path.join(args.resume_path, "offline_policy.pth")
        online_policy.load_state_dict(torch.load(online_policy_path, map_location=args.device))
        offline_policy.load_state_dict(torch.load(offline_policy_path, map_location=args.device))
        print(f"[{datetime.datetime.now()}] Loaded agent from: {args.resume_path}", flush=True)
   
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack
    )
    # collector
    if args.of4on_type == "DirectCopy":
        online_train_collector = Collector(online_policy, train_envs, buffer, exploration_noise=True)
    elif args.of4on_type == "DualCollect":
        # The default value of the probabiliyt of using online policy for experience collection is 1.0. i.e., only use the online policy
        online_train_collector = Of4OnCollector(
            online_policy, offline_policy, # two policies to select for collection based on a ratio
            online_policy, # initial collecting policy
            train_envs, buffer, exploration_noise=True)
    else:
        raise f"Unsupported of4on type: {args.of4on_type}"
    online_test_collector  = Collector(online_policy, test_envs, exploration_noise=True)
    offline_test_collector = Collector(offline_policy, test_envs_of, exploration_noise=True)

    # logging
    def create_logger(log_name, experiment_config):
        log_path = os.path.join(experiment_config.logdir, log_name)
        if experiment_config.logger == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=experiment_config.resume_id,
                config=experiment_config,
                project=experiment_config.wandb_project,
            )
        writer = SummaryWriter(log_path)

        writer.add_text("args", str(experiment_config))
        if experiment_config.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        else:  # wandb
            logger.load(writer)
        return logger

    online_log_name = os.path.join(log_name_prefix, "online")
    # logger for online learning
    online_logger = create_logger(online_log_name, args)
    # used to bootstrap offline learning in each phase
    current_best_online_policy_path = os.path.join(online_logger.writer.log_dir, "online_policy.pth")

    # Hook functions for training and testing
    def save_best_fn(policy, policy_type:str, log_path:str):
        policy_filename = f"{policy_type}_policy.pth"
        try:
            torch.save(policy.state_dict(), os.path.join(log_path, policy_filename))
        except:
            print(f"[{datetime.datetime.now()}] Failed to save {policy_type} policy in {log_path}", flush=True)
    save_best_online_fn = partial(save_best_fn, policy_type="online", log_path=online_logger.writer.log_dir)

    def stop_online_fn(mean_rewards):
        if args.early_stop:
            if env.spec.reward_threshold:
                return mean_rewards >= env.spec.reward_threshold
            elif "Pong" in args.task:
                return mean_rewards >= 20
            else:
                return False
        else:
            return False

    def stop_offline_fn(mean_rewards):
        return False
    
    def train_online_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= args.exploration_duration:
            eps = args.eps_train - env_step / args.exploration_duration * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        online_policy.set_eps(eps)
        if env_step % 1000 == 0:
            online_logger.write("train/env_step", env_step, {"train/eps": eps})

    # Workaround for saving the 'epoch', 'env_step' and 'gradient_step' in the logger for the next phase
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        return False

    def test_fn(epoch, env_step, policy_type:str):
        if policy_type == "online":
            policy = online_policy
        elif policy_type == "offline":
            policy = offline_policy
        else:
            raise f"Unknown policy type: {policy_type}"
        policy.set_eps(args.eps_test)
    test_online_fn = partial(test_fn, policy_type="online")
    test_offline_fn = partial(test_fn, policy_type="offline")

    # watch agent's performance
    def watch(policy, test_collector, policy_type:str):
        print("Setup test envs ...", flush=True)
        policy.eval()
        policy.set_eps(args.eps_test)
        #test_envs.seed(args.seed)
        if args.save_buffer_name and policy_type == "online":
            print(f"[{datetime.datetime.now()}] Generate buffer with size: {args.buffer_size}", flush=True)
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"[{datetime.datetime.now()}] Save buffer into: {args.save_buffer_name}", flush=True)
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print(f"[{datetime.datetime.now()}] Testing agent ...", flush=True)
            test_collector.reset()
            result = test_collector.collect(
                n_episode=args.test_num, render=args.render
            )
        rew = result["rews"].mean()
        print(f"[{datetime.datetime.now()}] Mean reward (over {result['n/ep']} episodes): {rew}", flush=True)

    if args.watch: # just watch the trained online policy, but not training
        watch(online_policy, online_test_collector, policy_type="online")
        watch(offline_policy, offline_test_collector, policy_type="offline")
        exit(0)

  
    # phase_epoch:
    #           index: phase number-1, value: number of online-learning epochs for the phase
    #           [20, 10, 10, 10]: 4 phases, 20 epochs for the first phase, 10 epochs for the rest
    #   Each phase starts with online learning, ends with offline learning
    #   Sarting the second phase, each online learning policy will be boostraped by the offline learning policy in the previous phase
    #   When the online learning finishes in the phase, the offline learning starts with the current replay buffer.
    #   The offline learning policy can be boostraped by the online learning policy in the current phase.
    #   Previous values of phase_epochs in manually setting: [5, 5, 5], [20, 10, 10, 10], [2, 1, 1, 1], [3,3], [5, 5, 5, 5]
    epoch_per_phase = args.online_epoch // args.num_phases
    phase_epochs = [epoch_per_phase] * args.num_phases
    # if online_epoch is divisible by num_phases, add the residual to the first phase
    if args.online_epoch%args.num_phases != 0:
        phase_epochs[0] = phase_epochs[0] + (args.online_epoch%args.num_phases)
    print(f"[{datetime.datetime.now()}] number of epochs in each phase: {phase_epochs}")

    '''
    # Schedule online-policy collecting ratio using phase as the step (except phase 1)
    ratio_step = abs(args.online_policy_collecting_ratio_start - args.online_policy_collecting_ratio_final) / (args.num_phases-1-1)
    online_policy_collecting_ratios = [1.0] # the first phase only uses online policy for collection
    online_policy_collecting_ratios += [args.online_policy_collecting_ratio + ratio_step*i for i in range(args.num_phases-1)]
    print(f"[{datetime.datetime.now()}] Online policy collecting ratios: {online_policy_collecting_ratios}", flush=True)
    '''
    # Initialize the experience collection setting of online policy
    offline_policy_performance = 0.0 # in the first phase, only use online policy for experience collection, so we can ignore it by setting it to 0.0
    offline_policy_performance_std = 0.0

    if args.of4on_type == "DualCollect": 
        online_policy_experience_collection_setting = OnlinePolicyExperienceCollectionSetting("Fixed", 1.0)
        print(f"[{datetime.datetime.now()}] Initialized experience-collection setting of online policy: {online_policy_experience_collection_setting}. It is used by the first phase. For phases starting from phase two, the user input will then update the setting accordingly.", flush=True)

    # pre-collect at least 50000 transitions with random action before training
    # replay_buffer_min_size = 50000
    print(f"[{datetime.datetime.now()}] Initial exploration of phase 1 using random policy: collecting {args.replay_buffer_min_size} transitions...", flush=True)
    # only use the online policy for the initial exploration
    online_train_collector.collect(n_step=args.replay_buffer_min_size, random=True)

    phase_max_epoch = 0
    previous_phase_best_offline_policy_path = ""
    performance_of_transferred_offline_policies = {}
    resume_from_log = False
    test_performance_of_online_policy_to_resume = None
    test_performance_of_offline_policy_to_transfer = None
    for idx, phase_epoch in enumerate(phase_epochs):
        phase_id = idx + 1 # phase id starts from 1

        # [online training ]
        print(f"[{datetime.datetime.now()}] Start phase {phase_id} online training (epoch: {phase_max_epoch} - {phase_max_epoch+phase_epoch})...", flush=True)
        
        # update the maximum number of epochs that the current phase is expected to reach
        # the actuall number of epochs for the current phase is phase_epoch
        phase_max_epoch += phase_epoch

        # set the ratio of using online policy for experience collection phase by phase
        # print(f"[{datetime.datetime.now()}] Set online policy collecting ratio: {online_policy_collecting_ratios[idx]}", flush=True)

        if phase_id > 1:
            resume_from_log = True
            # Directly copy the transferred offline policy to the online policy,
            # and initialize the buffer accordingly if it is requested be reset.
            if args.of4on_type == "DirectCopy":
                if args.reset_replay_buffer_per_phase:
                    print(f"[{datetime.datetime.now()}] Reset replay buffer for the phase {phase_id}...", flush=True)
                    if args.random_exploration_before_each_phase:
                        exploration_policy_name = "random policy"
                    else:
                        exploration_policy_name = f"online policy of phase {phase_id-1}"
                    print(f"[{datetime.datetime.now()}] Initial exploration of phase {phase_id} using {exploration_policy_name}: collecting {args.replay_buffer_min_size} transitions...", flush=True)
                    online_train_collector.collect(
                            n_step=args.replay_buffer_min_size,
                            random=args.random_exploration_before_each_phase)
                # a fresh new optimizer is created inside the function call.
                if args.transfer_best_offline_policy:
                    online_policy = create_online_policy(previous_phase_best_offline_policy_path)
                else:# bootstrap online policy with offline policy learned in the previous phase
                    online_policy.load_state_dict(offline_policy.state_dict())
                    online_policy.sync_weight()
                    online_policy.optim = torch.optim.Adam(
                            online_policy.model.parameters(), lr=args.online_lr, eps=0.01/32)
                # When creating a new online_policy instance, the follwing relinkings are needed
                online_train_collector.policy = online_policy
                online_test_collector.policy = online_policy
                transferred_policy_tag = "best" if args.transfer_best_offline_policy else "recent"
                print(f"[{datetime.datetime.now()}] Directly copied the transferred {transferred_policy_tag} offline policy to the online policy.")
            # Update the setting of experience collection by DualCollect for the next phase
            elif args.of4on_type == "DualCollect":
                if args.online_policy_collecting_setting_type == "Adaptive":
                    online_policy_collecting_setting_value = offline_policy_performance
                elif args.online_policy_collecting_setting_type == "Scheduling": # scheduling inside each phase and use epoch as the step
                    online_policy_collecting_setting_value = [args.online_policy_collecting_ratio_start, args.online_policy_collecting_ratio_final]
                elif args.online_policy_collecting_setting_type == "Fixed":
                    online_policy_collecting_setting_value = args.online_policy_collecting_ratio
                else:
                    assert False, f"Unexpected raito type of experience collection by online policy"
                online_policy_experience_collection_setting.setting_type = args.online_policy_collecting_setting_type
                online_policy_experience_collection_setting.setting_value = online_policy_collecting_setting_value
                print(f"[{datetime.datetime.now()}] Updated experience-collection setting of DualCollect for the next phase, phase {phase_id}: {online_policy_experience_collection_setting}.", flush=True)


        # (continue) online training
        if args.of4on_type == "DirectCopy":
            online_training_result = offpolicy_trainer(
                online_policy,
                online_train_collector,
                online_test_collector,
                phase_max_epoch, # args.online_epoch,
                args.online_step_per_epoch,
                args.online_step_per_collect,
                args.test_num,
                args.online_batch_size,
                train_fn=train_online_fn,
                test_fn=test_online_fn,
                stop_fn=stop_online_fn,
                save_best_fn=save_best_online_fn,
                save_checkpoint_fn=save_checkpoint_fn,
                resume_from_log = resume_from_log, # True for continuing the training from the previous phase
                logger=online_logger,
                update_per_step=args.online_update_per_step,
                test_in_train=False,
                show_progress=args.show_progress,
                test_performance_of_initial_policy_from_log = test_performance_of_offline_policy_to_transfer
            )
        elif args.of4on_type == "DualCollect":
            online_training_result = offpolicy_dualpolicycollection_trainer(
                online_policy,
                online_train_collector,
                online_test_collector,
                phase_max_epoch, # args.online_epoch,
                args.online_step_per_epoch,
                args.online_step_per_collect,
                args.test_num,
                args.online_batch_size,
                train_fn=train_online_fn,
                test_fn=test_online_fn,
                stop_fn=stop_online_fn,
                save_best_fn=save_best_online_fn,
                save_checkpoint_fn=save_checkpoint_fn,
                resume_from_log = resume_from_log, # True for continuing the training from the previous phase
                logger=online_logger,
                update_per_step=args.online_update_per_step,
                test_in_train=False,
                show_progress=args.show_progress,
                test_performance_of_initial_policy_from_log = test_performance_of_online_policy_to_resume,
                online_policy_experience_collection_setting = online_policy_experience_collection_setting,
            )

        # save and use it in the continual online training in the next phase
        test_performance_of_online_policy_to_resume = {}
        test_performance_of_online_policy_to_resume["rew"] = online_training_result["recent_reward"]
        test_performance_of_online_policy_to_resume["rew_std"] = online_training_result["recent_reward_std"]

        print(f"[{datetime.datetime.now()}] Finish phase {phase_id} online training ...", flush=True)
        pprint.pprint(online_training_result)
        sys.stdout.flush()


        # [offline training]
        print(f"[{datetime.datetime.now()}] Start phase {phase_id} offline training ...", flush=True)
        print(f"[{datetime.datetime.now()}] Current replay buffer size: {len(buffer)}", flush=True)
        # logger for offline learning
        offline_log_name = os.path.join(log_name_prefix, f"offline_{phase_id}")
        offline_logger = create_logger(offline_log_name, args)
        save_best_offline_fn = partial(save_best_fn, policy_type="offline", log_path=offline_logger.writer.log_dir)

        # Initialize a new offline policy instance for each phase
        offline_policy = create_offline_policy(current_best_online_policy_path)
        offline_test_collector.policy = offline_policy

        if args.offline_epoch_setting == 1: # 5X of gradient steps of online learning in the current phase
            offline_epoch = int(phase_epoch * args.online_step_per_epoch * args.online_update_per_step * 5 / args.offline_update_per_epoch)
        elif args.offline_epoch_setting == 2: # 5X of gradient steps of online learning inferred by the current buffer
            offline_epoch = int(len(buffer) * args.online_update_per_step * 5 / args.offline_update_per_epoch)
        else:
            offline_epoch = args.offline_epoch
        print(f"[{datetime.datetime.now()}] Phase {phase_id} offline learning epochs: {offline_epoch}", flush=True)

        offline_training_result = offline_trainer(
            offline_policy,
            buffer,
            offline_test_collector,
            offline_epoch, #args.offline_epoch,
            args.offline_update_per_epoch,
            args.test_num,
            args.offline_batch_size,
            stop_fn=stop_offline_fn,
            save_best_fn=save_best_offline_fn,
            logger=offline_logger,
            show_progress=args.show_progress,
        )

        # Update the offline policy inside the online training collector for the next phase
        # No need to sync the target network with the online network here since only the 
        # online network is used to collect experience when using the offline policy.
        test_performance_of_offline_policy_to_transfer = {}
        if args.transfer_best_offline_policy:
            phase_best_offline_policy_path = os.path.join(offline_logger.writer.log_dir, "offline_policy.pth")
            if args.of4on_type == "DualCollect":
                online_train_collector.offline_policy.load_state_dict(
                    torch.load(phase_best_offline_policy_path, map_location=args.device)
                )
            offline_policy_performance = float(offline_training_result["best_reward"])
            offline_policy_performance_std = offline_training_result["best_reward_std"]
        else:
            if args.of4on_type == "DualCollect":
                online_train_collector.offline_policy = offline_policy
            offline_policy_performance = float(offline_training_result["recent_reward"])
            offline_policy_performance_std = offline_training_result["recent_reward_std"]
        test_performance_of_offline_policy_to_transfer["rew"] = offline_policy_performance
        test_performance_of_offline_policy_to_transfer["rew_std"] = offline_policy_performance_std
        performance_of_transferred_offline_policies[phase_max_epoch*args.online_step_per_epoch] = offline_policy_performance

        print(f"[{datetime.datetime.now()}] Finish phase {phase_id} offline training ...", flush=True)
        pprint.pprint(offline_training_result)
        sys.stdout.flush()


    with open(os.path.join(experiment_result_dir, "performance_of_transferred_offline_policies.json"), "w") as f:
        json.dump(performance_of_transferred_offline_policies, f, indent=4)

    online_policy_test_rewards = online_logger.retrieve_info_from_log("test/reward")
    with open(os.path.join(experiment_result_dir, "online_policy_test_rewards.json"), "w") as f:
        json.dump(online_policy_test_rewards, f, indent=4)

if __name__ == "__main__":
    test_of4on(get_args())
