import os
import torch
import numpy as np
import random

def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def total_reseed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)