# utils/device.py
# -*- encoding: utf-8 -*-

import torch

def get_compute_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_gpu_cache():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True