# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from crf import CRF
from er import EntropyRegularization


class S2S(nn.Module):
    def __init__(self, window_len, out_len):
        super(S2S, self).__init__()
        self.conv1 = nn.Conv1d(1, 30, 11, padding=5)  # 10
        self.conv2 = nn.Conv1d(30, 30, 9, padding=4)  # 8
        self.conv3 = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4 = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5 = nn.Conv1d(50, 50, 5, padding=2)
        self.fc1 = nn.Linear(50 * window_len, 1024)
        self.fc2 = nn.Linear(1024, out_len)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class S2P_on(nn.Module):
    def __init__(self, window_len):
        super(S2P_on, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_p = nn.Conv1d(30, 40, 9, padding=4)  # 6
        self.conv4_p = nn.Conv1d(40, 50, 7, padding=3)
        self.conv5_p = nn.Conv1d(50, 50, 5, padding=2)
        # self.conv6_p = nn.Conv1d(50, 50, 5, padding=2)
        self.fc1_p = nn.Linear(50 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, 1)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_s = nn.Conv1d(30, 40, 9, padding=4)  # 6
        self.conv4_s = nn.Conv1d(40, 50, 7, padding=3)
        self.conv5_s = nn.Conv1d(50, 50, 5, padding=2)
        # self.conv6_s = nn.Conv1d(50, 50, 5, padding=2)
        self.fc1_s = nn.Linear(50 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        # x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        # y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = torch.sigmoid(self.fc2_s(y))

        return x, y


class S2P_State(nn.Module):
    def __init__(self, window_len, state_num):
        super(S2P_State, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_p = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_p = nn.Linear(60 * window_len, 1024)
        # self.fc2_p = nn.Linear(512,32)
        self.fc3_p = nn.Linear(1024, 1)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        # self.fc2_s = nn.Linear(512,32)
        self.fc3_s = nn.Linear(1024, state_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        # x = F.relu(self.fc2_p(x))
        x = self.fc3_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        # y = F.relu(self.fc2_s(y))
        y = self.fc3_s(y)

        return x, y


class S2P_State2(nn.Module):
    def __init__(self, window_len, state_num):
        super(S2P_State2, self).__init__()
        self.conv11_p = nn.Conv1d(1, 50, 11, padding=5)  # 10
        self.conv12_p = nn.Conv1d(1, 50, 9, padding=4)  # 8
        self.conv13_p = nn.Conv1d(1, 50, 7, padding=3)  # 6
        self.conv2_p = nn.Conv1d(150, 50, 5, padding=2)
        self.fc1_p = nn.Linear(50 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, state_num)

        self.conv11_s = nn.Conv1d(1, 50, 11, padding=5)  # 10
        self.conv12_s = nn.Conv1d(1, 50, 9, padding=4)  # 8
        self.conv13_s = nn.Conv1d(1, 50, 7, padding=3)  # 6
        self.conv2_s = nn.Conv1d(150, 50, 5, padding=2)
        self.fc1_s = nn.Linear(50 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, state_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x1 = F.relu(self.conv11_p(x))
        x2 = F.relu(self.conv12_p(x))
        x3 = F.relu(self.conv13_p(x))
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.conv2_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)

        y1 = F.relu(self.conv11_s(y))
        y2 = F.relu(self.conv12_s(y))
        y3 = F.relu(self.conv13_s(y))
        y = torch.cat([y1, y2, y3], dim=1)
        y = F.relu(self.conv2_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = self.fc2_s(y)

        return x, y


class S2P_State_a(nn.Module):
    def __init__(self, window_len, state_num):
        super(S2P_State_a, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_p = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_p = nn.Linear(60 * window_len, 1024)
        # self.fc2_p = nn.Linear(512,32)
        self.fc3_p = nn.Linear(1024, state_num)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)  # 10
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)  # 8
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        # self.fc2_s = nn.Linear(512,32)
        self.fc3_s = nn.Linear(1024, state_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        # x = F.relu(self.fc2_p(x))
        x = self.fc3_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        # y = F.relu(self.fc2_s(y))
        y = self.fc3_s(y)

        return x, y


# on-off state
class S2S_on(nn.Module):
    def __init__(self, window_len, out_len):
        super(S2S_on, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 11, padding=5)  # 10
        self.conv2_p = nn.Conv1d(30, 30, 9, padding=4)  # 8
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 50, 5, padding=2)
        self.conv6_p = nn.Conv1d(50, 50, 5, padding=2)
        self.fc1_p = nn.Linear(50 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, out_len)

        self.conv1_s = nn.Conv1d(1, 30, 11, padding=5)  # 10
        self.conv2_s = nn.Conv1d(30, 30, 9, padding=4)  # 8
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)  # 6
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 50, 5, padding=2)
        self.conv6_s = nn.Conv1d(50, 50, 5, padding=2)
        self.fc1_s = nn.Linear(50 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, out_len)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = torch.sigmoid(self.fc2_s(y))

        return x, y


class S2S_state(nn.Module):
    def __init__(self, window_len, out_len, state_num):
        super(S2S_state, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_p = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_p = nn.Linear(60 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, out_len * state_num)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, out_len * state_num)

        self.crf = CRF(num_tags=state_num, batch_first=True)

    def calc_crf_loss(self, out, tgt):
        return -self.crf(out, tgt, reduction='mean')

    def decode(self, out):
        return self.crf.decode(out)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        # x.shape=[batchsize,]
        x = x.flatten(-2, -1)
        #         print('x.shape', x.shape)
        x = F.relu(self.fc1_p(x))
        #         print('x.shape', x.shape)
        x = self.fc2_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = self.fc2_s(y)

        return x, y


class S2S_state_sin_crf(nn.Module):
    def __init__(self, window_len, out_len, state_num):
        super(S2S_state_sin_crf, self).__init__()
        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, out_len * state_num)

    def decode(self, out):
        return self.crf.decode(out)

    def forward(self, y):
        y = y.unsqueeze(1)
        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = self.fc2_s(y)

        return y
        '''
class S2P_State_a(nn.Module):
    def __init__(self, window_len, state_num):
        super(S2P_State_a), self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)#10
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)#8
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)#6
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_p = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_p = nn.Linear(60 * window_len, 1024)
        #self.fc2_p = nn.Linear(512,32)
        self.fc3_p = nn.Linear(1024, state_num)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)#10
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)#8
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)#6
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        #self.fc2_s = nn.Linear(512,32)
        self.fc3_s = nn.Linear(1024, state_num)

        self.crf = CRF(num_tags=state_num, batch_first=True)

    def calc_crf_loss(self, out, tgt):
        return -self.crf(out, tgt, reduction='mean')

    def decode(self, out):
        return self.crf.decode(out)    

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        #x = F.relu(self.fc2_p(x))
        x = self.fc3_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        #y = F.relu(self.fc2_s(y))
        y = self.fc3_s(y)

        return x, y
        '''


class S2S_state_(nn.Module):
    def __init__(self, window_len, out_len, state_num):
        super(S2S_state_, self).__init__()
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_p = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_p = nn.Linear(60 * window_len, 1024)
        self.fc2_p = nn.Linear(1024, out_len * state_num)

        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_len, 1024)
        self.fc2_s = nn.Linear(1024, out_len * state_num)

        self.crf = CRF(num_tags=state_num, batch_first=True)
        self.er = EntropyRegularization()

    def calc_crf_loss(self, out, tgt):
        return -self.crf(out, tgt, reduction='mean')

    def calc_er_loss(self, out):
        return self.er(out)

    def decode(self, out):
        return self.crf.decode(out)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = x
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        x = self.fc2_p(x)

        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        y = self.fc2_s(y)

        return x, y




