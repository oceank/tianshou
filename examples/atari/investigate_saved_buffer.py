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

from tianshou.data import ReplayBuffer

def extract_top_x_percent_episode(buffer, top_x_percent, buffer_size=1000000, frame_stack=4):
    # calculate the supporting information for the selection of top episodes
    top_x_percent = 0.1
    ep_done_indices = np.where(buffer.done==True)[0]
    ep_start_indices = np.zeros_like(ep_done_indices)
    ep_start_indices[1:] = ep_done_indices[:-1]
    ep_returns = np.array([buffer.rew[s:(1+e)].sum() for s, e in zip(ep_start_indices, ep_done_indices)])
    sorted_ep_indices = np.argsort(ep_returns, kind='stable')
    num_sel_episodes = int(len(ep_done_indices)*top_x_percent)

    # create a new buffer for offline
    sel_buffer = ReplayBuffer(size=buffer_size, stack_num=frame_stack, ignore_obs_next=True, save_only_last_obs=True)

    # Add selected episodes into the newly recreated buffer
    for sel_ep_idx in sorted_ep_indices[-num_sel_episodes:]:
        for transition_idx in range(ep_start_indices[sel_ep_idx], 1+ep_done_indices[sel_ep_idx]):
            sel_buffer.add(buffer[transition_idx])
    
    return sel_buffer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-num", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--saved-buffer-filepath", type=str, default=None, help="Path to saved buffer")
    return parser.parse_args()

def investigate_buffer(args=get_args()):
    if args.saved_buffer_filepath is None:
        raise ValueError("Please specify path to saved buffer")
    
    '''
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=args.training_num,
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack
    )
    '''

    try:
        print(f"Loading the saved buffer")
        buffer = VectorReplayBuffer.load_hdf5(args.saved_buffer_filepath)
    except Exception as e:
        print(e)
    
    print(f"Buffer loaded: {len(buffer)} samples")
    top_x_percent_list = [0.1, 0.2, 0.5]
    for top_x_percent in top_x_percent_list:
        sel_buffer = extract_top_x_percent_episode(buffer, top_x_percent)
        print(f"[Top {top_x_percent*100}%100]: Extracted {len(sel_buffer)} samples")
        sel_path = args.saved_buffer_filepath.replace(".hdf5", f"_top{int(top_x_percent*100)}.hdf5")
        sel_buffer.save_hdf5(sel_path, compression="gzip")
    print("Done")

if __name__ == "__main__":
    investigate_buffer(get_args())