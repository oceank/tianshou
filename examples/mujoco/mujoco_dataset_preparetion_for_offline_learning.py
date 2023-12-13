#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
import json

import numpy as np
import torch
from mujoco_env import make_mujoco_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ant-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=False, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--expr-time", type=str, default="000000-000000")
    parser.add_argument("--checkpoint-step", type=int, default=200000)
    parser.add_argument("--step-per-epoch", type=int, default=5000)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--show-progress", default=False, action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=4)

    return parser.parse_args()

def create_or_load_policy(args, checkpoint_policy_filepath):
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(args.action_shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=args.action_space,
    )

    # load a previous policy
    checkpoint = torch.load(checkpoint_policy_filepath, map_location=args.device)
    policy.load_state_dict(checkpoint["model"])
    print("Loaded agent from: ", checkpoint_policy_filepath)

    return policy

def collect_experience(args=get_args()):
    # threads for pytorch
    torch.set_num_threads(args.torch_num_threads)

    # environments for experience collection
    print(f"[Create environments for the behavior policy to collect new experiences]")
    env, train_envs, test_envs = make_mujoco_env(
        args.task, args.seed, args.training_num, args.test_num, obs_norm=False
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.action_space = env.action_space
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # experiment result directory
    args.algo_name = "sac"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), args.expr_time)
    log_path = os.path.join(args.logdir, log_name)
    print(f"\n[Experiment Directory: {log_path}]")

    # load the saved buffer and initialize the new buffer
    print(f"\n[Load the previously saved replay buffer and initialize the new buffer with the first {args.checkpoint_step} experiences]")
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        new_buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
        new_buffer = ReplayBuffer(args.buffer_size)
    buffer_filepath = os.path.join(log_path, f"buffer.hdf5")
    buffer = buffer.load_hdf5(buffer_filepath) # load onto RAM
    steps_of_experiences_to_copy=args.checkpoint_step # 200k, 40 epochs
    for experience_idx in range(steps_of_experiences_to_copy):
        new_buffer.add(buffer[experience_idx])
        if (experience_idx+1)%args.step_per_epoch == 0:
            print(f"===> New buffer: loaded {experience_idx+1} experiences")
    #new_buffer.add(buffer[0:steps_of_experiences_to_copy])
    print(f"The new buffer is initialized with {len(new_buffer)} experiences")

    # load the policy for experience collection
    checkpoint_epoch = args.checkpoint_step//args.step_per_epoch
    checkpoint_policy_filename = f"checkpoint_epoch{checkpoint_epoch}.pth"
    checkpoint_policy_filepath = os.path.join(log_path, checkpoint_policy_filename)
    print(f"\n[Loading the behavior policy at {checkpoint_policy_filepath}]")
    policy = create_or_load_policy(args, checkpoint_policy_filepath)
    print(f"The behavior policy is loaded")

    # collect new experiences and save to file
    use_random_policy = (args.checkpoint_step==0)
    behavior_policy_name = "random_policy" if use_random_policy else checkpoint_policy_filename
    print(f"\n[Collecting new experienes using the behavior policy, {behavior_policy_name}]")
    collector = Collector(policy, train_envs, new_buffer)
    steps_of_experiences_to_collect = args.buffer_size - steps_of_experiences_to_copy
    collector.collect(n_step=steps_of_experiences_to_collect, progress_interval=args.step_per_epoch, random=use_random_policy)
    #collector.collect(n_step=args.start_timesteps, random=True)
    print(f"Collected new {steps_of_experiences_to_collect} experiences and the new buffer has a size of {len(new_buffer)}")
    # Save the new buffer to file
    buffer_filepath = os.path.join(args.logdir, log_name, f"buffer_oldSaved{steps_of_experiences_to_copy}_newCollected{steps_of_experiences_to_collect}.hdf5")
    new_buffer.save_hdf5(buffer_filepath, compression="gzip")
    print(f"New buffer is saved to file")



if __name__ == "__main__":
    collect_experience()
