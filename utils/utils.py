import torch
import numpy as np
import random
import pyro

""" This file contains various utilities that don't fit the scope of the other utility files.
"""


def set_seed(seed):
    """ Sets the random seed for various modules to (hopefully) get deterministic behavior

    See further: https://github.com/pytorch/pytorch/issues/7068#issuecomment-484918113
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pyro.set_rng_seed(seed)
