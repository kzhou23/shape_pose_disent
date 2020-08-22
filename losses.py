import torch
import torch.nn as nn

def l1_loss(x, y):
    l1_loss = nn.L1Loss()
    loss = l1_loss(x, y)
    return loss