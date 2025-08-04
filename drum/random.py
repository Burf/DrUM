import os
import random

import numpy as np
import torch

def set_python_seed(seed = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_random_seed(seed = 0):
    random.seed(seed)

def set_numpy_seed(seed = 0):
    np.random.seed(seed)
    
def set_torch_seed(seed = 0, deterministic = True, benchmark = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
            
def set_seed(seed = 0, deterministic = True, benchmark = False):
    set_python_seed(seed)
    set_random_seed(seed)
    set_numpy_seed(seed)
    set_torch_seed(seed, deterministic = deterministic, benchmark = benchmark)