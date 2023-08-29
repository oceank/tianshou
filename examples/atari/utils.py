import os
import random
import torch
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


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
