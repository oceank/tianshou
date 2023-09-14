import os
import random
import torch
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

# Returns of Random Agent and Human Beginner
games = ["Asterix", "Gravitar", "NameThisGame", "Pong", "Qbert", "Seaquest"]
returns_random_agent = [210.0, 173.0, 2292.3, -20.7, 163.9, 68.4]
returns_human_beginner = [8603.3, 3351.4, 8049.0, 14.6, 13455.0, 42054.7]
returns_random_agent_and_human = {}
for game, score_r, score_h in zip(games, returns_random_agent, returns_human_beginner):
    returns_random_agent_and_human[game] = {"random": score_r, "human": score_h}
'''
scores_random_agent_and_human = {
    "Asterix": {"random": 210.0, "human": 8603.3},
    "Gravitar": {"random": 173.0, "human": 3351.4},
    "NameThisGame": {"random": 2292.3, "human": 8049.0},
    "Pong": {"random": -20.7, "human": 14.6},
    "Qbert": {"random": 163.9, "human": 13455.0},
    "Seaquest": {"random": 68.4, "human": 42054.7},
    
}
'''

def cal_human_normalized_score(game, agent_return):
    recorded_scores = returns_random_agent_and_human[game]
    rr, hr = recorded_scores['random'], recorded_scores['human']
    return (agent_return - rr) / abs(hr - rr)

def set_torch_seed(SEED):
    torch.manual_seed(SEED)                         # Seed the RNG for all devices (both CPU and CUDA).
    torch.cuda.manual_seed(SEED)                    # Set a fixed seed for the current GPU.
    torch.cuda.manual_seed_all(SEED)                # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

def set_determenistic_mode(SEED, disable_cudnn):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)                               # Set python seed for custom operators.
    rs = RandomState(MT19937(SeedSequence(SEED)))   # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)
  
    set_torch_seed(SEED)
    if not disable_cudnn:
        torch.backends.cudnn.benchmark = False      # Causes cuDNN to deterministically select an algorithm,
                                                    # possibly at the cost of reduced performance
                                                    # (the algorithm itself may be nondeterministic).
        torch.backends.cudnn.deterministic = True   # Causes cuDNN to use a deterministic convolution algorithm,
                                                    # but may slow down performance.
                                                    # It will not guarantee that your training process is deterministic
                                                    # if you are using other libraries that may use nondeterministic algorithms.
    else:
        torch.backends.cudnn.enabled = False        # Controls whether cuDNN is enabled or not.
                                                    # If you want to enable cuDNN, set it to True.

