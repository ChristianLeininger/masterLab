import random
import numpy as np
import torch
import logging

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Additional seeding logic as required

    # Optionally, set determinism for deep learning operations
    #torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = Falsedef log_array(arr):

def log_array(arr):
    # Convert the array to a string with newline characters between rows
    array_str = '\n'.join([' '.join(map(str, row)) for row in arr])
    logging.info("\n" + array_str)