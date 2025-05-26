# utils/device.py
# -*- encoding: utf-8 -*-

import torch

def get_compute_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')