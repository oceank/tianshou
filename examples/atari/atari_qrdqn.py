import argparse
import datetime
import os
import sys
import pprint
import datetime
import random
import json

import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch
from atari_network import QRDQN
from atari_wrapper import make_atari_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import QRDQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import DataParallelNet

from examples.atari.utils import set_torch_seed, set_determenistic_mode, returns_random_agent_and_human, cal_human_normalized_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--train-env-num-threads", type=int, default=4)
    parser.add_argument("--test-env-num-threads", type=int, default=10)
    parser.add_argument("--task", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.01)
    parser.add_argument("--exploration-duration", type=int, default=1000000)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--replay-buffer-min-size", type=int, default=50000)
    parser.add_argument("--save-buffer", action="store_true")
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-quantiles", type=int, default=200)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--target-update-freq", type=int, default=5000)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--early-stop", type=bool, default=False)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=4)
    parser.add_argument("--update-per-step", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=4)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--use-multi-gpus", action="store_true")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
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
    parser.add_argument("--save-buffer-name", type=str, default=None)
    ## disable cudnn for avoding inconsistent randomness across machines
    parser.add_argument("--disable-cudnn", action="store_true")
    return parser.parse_args()


def test_qrdqn(args=get_args()):
    torch.set_num_threads(args.torch_num_threads)

    # create directory for logging
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "qrdqn"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    os.makedirs(log_path, exist_ok=True)

    # save experiment configuration
    print(f"[{datetime.datetime.now()}] experiment configuration:", flush=True)
    pprint.pprint(vars(args), indent=4)
    sys.stdout.flush()
    print(f"[{datetime.datetime.now()}] The available number of GPUs: {torch.cuda.device_count()}", flush=True)
    with open(os.path.join(log_path, "traning_configuration.json"), "w") as f:
        print(vars(args))
        json.dump(vars(args), f, indent=4)

    # seed
    disable_cudnn = args.disable_cudnn
    set_determenistic_mode(args.seed, disable_cudnn)

    # create envs
    env, train_envs, test_envs = make_atari_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        train_env_num_threads = args.train_env_num_threads,
        test_env_num_threads = args.test_env_num_threads,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    env.action_space.seed(args.seed)
    train_envs.action_space.seed(args.seed)
    test_envs.action_space.seed(args.seed)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    game = args.task.replace("NoFrameskip-v4", "")
    recorded_scores = returns_random_agent_and_human[game]
    print(f"[{datetime.datetime.now()}] Game {game}: Random Agent ({recorded_scores['random']}), Human ({recorded_scores['human']})")
    print(f"[{datetime.datetime.now()}] Observations shape: {args.state_shape}", flush=True)
    print(f"[{datetime.datetime.now()}] Actions shape: {args.action_shape}", flush=True)


    # define model
    set_torch_seed(args.seed)
    if torch.cuda.is_available and args.use_multi_gpus:
        assert torch.cuda.device_count() > 1, f"The available number of GPUs ({torch.cuda.device_count()}) < 2"    
        net = DataParallelNet(QRDQN(*args.state_shape, args.action_shape, args.num_quantiles, device=None)).to(args.device)
    else:
        net = QRDQN(*args.state_shape, args.action_shape, args.num_quantiles, args.device)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr, eps=0.01/32)
    # define policy
    policy = QRDQNPolicy(
        net,
        optim,
        args.gamma,
        args.num_quantiles,
        args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
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
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)

    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer, game=game)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        if args.early_stop:
            if env.spec.reward_threshold:
                return mean_rewards >= env.spec.reward_threshold
            elif "Pong" in args.task:
                return mean_rewards >= 20
            else:
                return False
        else:
            return False

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= args.exploration_duration:
            eps = args.eps_train - env_step / args.exploration_duration * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        print(f"[{datetime.datetime.now()}] Setup test envs ...", flush=True)
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
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
            result['hnss'] = cal_human_normalized_score(logger.game, result["rews"])
            result['hns'] = result['hnss'].mean()
            result['hns_std'] = result['hnss'].std()
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

    if args.watch:
        watch()
        exit(0)

    np.random.seed(args.seed)
    # pre-collect at least 50000 transitions with random action before training
    # replay_buffer_min_size = 50000
    train_collector.collect(n_step=args.replay_buffer_min_size, random=True)

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
        show_progress=args.show_progress,
    )

    print(f"[{datetime.datetime.now()}] Finish training ...", flush=True)
    pprint.pprint(result)
    sys.stdout.flush()
    #watch()

    if args.save_buffer:
        buffer_filepath = os.path.join(args.logdir, log_name, f"buffer_{len(buffer)}.hdf5")
        buffer.save_hdf5(buffer_filepath)

    online_policy_test_rewards = logger.retrieve_info_from_log("test/reward")
    with open(os.path.join(args.logdir, log_name, "online_policy_test_rewards.json"), "w") as f:
        json.dump(online_policy_test_rewards, f, indent=4)

    online_policy_test_hnss = logger.retrieve_info_from_log("test/hns")
    with open(os.path.join(args.logdir, log_name, "online_policy_test_hnss.json"), "w") as f:
        json.dump(online_policy_test_hnss, f, indent=4)

if __name__ == "__main__":
    test_qrdqn(get_args())
