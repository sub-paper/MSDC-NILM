import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropyRegularization(nn.Module):
    def __init__(self,):
        super(EntropyRegularization, self).__init__()
        self.eps = 1e-7

    def forward(self, x):
        '''
        x: batch_size x window_len x state_num
        return: batch_size x window_len
        '''
        x = F.softmax(x, dim=-1)
        return torch.sum(x * torch.log(x + self.eps), dim=-1).mean()