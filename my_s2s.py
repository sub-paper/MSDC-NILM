# -*- coding: utf-8 -*-

import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import data_provider
import model

# our schme multiple states without CRF layer
params_appliance = {
    'kettle': {
        'window_len': 599,
        'uk_on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        'uk_state_num': 2,
        'uk_state': [2000, 4500],
        'uk_state_average': [1.15, 2280.79],  # 1.2230124 2796.673
        's2s_length': 128
    },
    'microwave': {
        'window_len': 599,
        'redd_on_power_threshold': 300,
        'uk_on_power_threshold': 300,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        'redd_state_num': 2,
        'redd_state': [300, 3000],
        'redd_state_average': [4.2, 1557.501],
        'uk_state_num': 2,
        'uk_state': [300, 3000],
        'uk_state_average': [1.4, 1551.3],
        's2s_length': 128
    },
    'fridge': {
        'window_len': 599,
        'redd_on_power_threshold': 50,
        'uk_on_power_threshold': 20,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        'redd_state_num': 3,
        'redd_state': [50, 300, 500],
        'redd_state_average': [3.2, 143.3, 397.3],
        'redd_house1_state_num': 3,
        'redd_house1_state': [50, 300, 500],
        'redd_house1_state_average': [6.49, 192.57, 443],
        'redd_house2_state_num': 3,
        'redd_house2_state': [50, 300, 500],
        'redd_house2_state_average': [6.34, 162.87, 418.36],
        'redd_house3_state_num': 3,
        'redd_house3_state': [50, 300, 500],
        'redd_house3_state_average': [0.54, 118.85, 409.75],
        'uk_state_num': 3,
        'uk_state': [20, 200, 2500],
        'uk_state_average': [0.13, 87.26, 246.5],
        's2s_length': 512
    },
    'dishwasher': {
        'window_len': 599,
        'redd_on_power_threshold': 150,
        'uk_on_power_threshold': 50,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        'redd_state_num': 4,
        'redd_state': [150, 300, 1000, 3000],
        'redd_state_average': [0.57, 232.91, 733.89, 1198.31],
        'redd_house1_state_num': 4,
        'redd_house1_state': [150, 300, 1000, 3000],
        'redd_house1_state_average': [0.21, 216.75, 438.51, 1105.08],
        'redd_house2_state_num': 3,
        'redd_house2_state': [150, 1000, 3000],
        'redd_house2_state_average': [0.16, 250.26, 1197.93],
        'redd_house3_state_num': 3,
        'redd_house3_state': [50, 400, 1000],
        'redd_house3_state_average': [0.97, 195.6, 743.42],
        'uk_state_num': 3,
        'uk_state': [50, 1000, 4500],
        'uk_state_average': [0.89, 122.56, 2324.9],
        's2s_length': 1536
    },
    'washingmachine': {
        'window_len': 599,
        'redd_on_power_threshold': 500,
        'uk_on_power_threshold': 50,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        'redd_state_num': 2,
        'redd_state': [500, 5000],
        'redd_state_average': [0, 2627.3],
        'uk_state_num': 3,
        'uk_state': [50, 800, 3500],
        'uk_state_average': [0.13, 204.64, 1892.85],
        'uk_house2_state_num': 4,
        'uk_house2_state': [50, 200, 1000, 4000],
        'uk_house2_state_average': [2.83, 114.34, 330.25, 2100.14],
        's2s_length': 2000
    },
}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance_name',
                        type=str,
                        default='dishwasher',
                        help='the name of target appliance')
    parser.add_argument('--data_dir',
                        type=str,
                        default='/redd/',
                        help='this is the directory of the training samples')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='The batch size of training examples')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=20,
                        help='The number of epoches.')
    parser.add_argument('--patience',
                        type=int,
                        default=1)
    parser.add_argument('--seed',
                        type=int,
                        default=819)
    return parser.parse_args()


args = get_arguments()
# save_path='/result/redd_fa_132_'+str(args.seed)+'_'
save_path = '/data/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_dataset():
    # 训练测试不在同一房间
    tra_x = args.data_dir + args.appliance_name + '_mains_' + 'tra_small'  # save path for mains
    val_x = args.data_dir + args.appliance_name + '_mains_' + 'val'

    tra_y = args.data_dir + args.appliance_name + '_tra_small' + '_' + 'pointnet'  # save path for target
    val_y = args.data_dir + args.appliance_name + '_val' + '_' + 'pointnet'

    tra_s = args.data_dir + args.appliance_name + '_tra_small' + '_' + 'pointnet_s'  # save path for target
    val_s = args.data_dir + args.appliance_name + '_val' + '_' + 'pointnet_s'

    test_x = args.data_dir + args.appliance_name + '_test_x'
    test_y = args.data_dir + args.appliance_name + '_test_gt'
    test_s = args.data_dir + args.appliance_name + '_test_gt_s'
     #单房间
    # tra_x = args.data_dir + args.appliance_name + 'house6' + '_mains_' + 'tra_small'  # save path for mains
    # val_x = args.data_dir + args.appliance_name + 'house6' + '_mains_' + 'val'
    #
    # tra_y = args.data_dir + args.appliance_name + 'house6' + '_tra_small' + '_' + 'pointnet'  # save path for target
    # val_y = args.data_dir + args.appliance_name + 'house6' + '_val' + '_' + 'pointnet'
    #
    #
    # tra_s = args.data_dir + args.appliance_name + 'house6' + '_tra_small' + '_' + 'pointnet_s'  # save path for target
    # val_s = args.data_dir + args.appliance_name + 'house6' + '_val' + '_' + 'pointnet_s'
    #
    # test_x = args.data_dir + args.appliance_name + 'house6' + '_test_x'
    # test_y = args.data_dir + args.appliance_name + 'house6' + '_test_gt'
    # test_s = args.data_dir + args.appliance_name + 'house6' + '_test_gt_s'

    tra_set_x = np.load(tra_x + '.npy').astype(np.float32)
    tra_set_y = np.load(tra_y + '.npy').astype(np.float32)
    tra_set_s = np.load(tra_s + '.npy').astype(np.float32)
    val_set_x = np.load(val_x + '.npy').astype(np.float32)
    val_set_y = np.load(val_y + '.npy').astype(np.float32)
    val_set_s = np.load(val_s + '.npy').astype(np.float32)
    test_set_x = np.load(test_x + '.npy').astype(np.float32)
    test_set_y = np.load(test_y + '.npy').astype(np.float32)
    test_set_s = np.load(test_s + '.npy').astype(np.float32)
    

    print('training set:', tra_set_x.shape, tra_set_y.shape, tra_set_s.shape)
    print('validation set:', val_set_x.shape, val_set_y.shape, val_set_s.shape)
    print('testing set:', test_set_x.shape, test_set_y.shape, test_set_s.shape)

    return tra_set_x, tra_set_y, tra_set_s, val_set_x, val_set_y, val_set_s, test_set_x, test_set_y, test_set_s


# load the data set
tra_set_x, tra_set_y, tra_set_s, val_set_x, val_set_y, val_set_s, test_set_x, test_set_y, test_set_s = load_dataset()

# hyper parameters according to appliance
window_len = 400
out_len = 64
state_num = 4  # params_appliance[args.appliance_name]['redd_house3_state_num']
print(state_num)
offset = int(0.5 * (window_len - 1.0))

tra_kwag = {
    'inputs': tra_set_x,
    'targets': tra_set_y,
    'targets_s': tra_set_s,
}
val_kwag = {
    'inputs': val_set_x,
    'targets': val_set_y,
    'targets_s': val_set_s,
}
test_kwag = {
    'inputs': test_set_x,
    'targets': test_set_y,
    'targets_s': test_set_s,
}
mean = params_appliance[args.appliance_name]['mean']
std = params_appliance[args.appliance_name]['std']
threshold = (params_appliance[args.appliance_name]['redd_on_power_threshold'] - mean) / std
tra_provider = data_provider.S2S_State_Slider(batch_size=args.batch_size,
                                              shuffle=True, offset=offset, length=window_len,
                                              out_len=out_len)  # , threshold=threshold
val_provider = data_provider.S2S_State_Slider(batch_size=5000,
                                              shuffle=False, offset=offset, length=window_len, out_len=out_len)
test_provider = data_provider.S2S_State_Slider(batch_size=5000,
                                               shuffle=False, offset=offset, length=window_len, out_len=out_len)

m = model.S2S_state(window_len, out_len, state_num).to(device)
_params = filter(lambda p: p.requires_grad, m.parameters())
optimizer = torch.optim.Adam(_params, lr=1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
mean = params_appliance[args.appliance_name]['mean']
std = params_appliance[args.appliance_name]['std']

# train & val
best_state_dict_path = 'state_dict/{}'.format(args.appliance_name)

best_val_loss = float('inf')
best_val_epoch = -1
for epoch in range(args.n_epoch):
    train_loss, n_batch_train = 0, 0
    for batch in tra_provider.feed(**tra_kwag):
        m.train()
        optimizer.zero_grad()
        x_train, y_train, s_train = batch
        # x_train.shape=[batch_size, window_size]
        # y_train.shape=[batch_size, out_len]
        # s_train.shape=[batch_size, out_len]
        x_train = torch.tensor(x_train, dtype=torch.float, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float, device=device)
        s_train = torch.tensor(s_train, dtype=torch.long, device=device)
        op_train, os_train = m(x_train)
        # op_train.shape = [batch_size, out_len * state_num]
        # os_train.shape = [batch_size, out_len * state_num]
        op_train = torch.reshape(op_train, (op_train.shape[0], out_len, state_num))
        os_train = torch.reshape(os_train, (os_train.shape[0], out_len, state_num))
        # op_train.shape = [batch_size, out_len, state_num]
        # os_train.shape = [batch_size, out_len, state_num]
        oss_train = F.softmax(os_train, dim=-1)
        # oss_train.shape = [batch_size, out_len, state_num]
        o_train = torch.sum(oss_train * op_train, dim=-1, keepdim=False)
        # o_train.shape = [batch_size*out_len, state_num]
        os_train = os_train.flatten(0, 1)
        s_train = s_train.flatten(0, 1)
        #         print('s_train for loss', s_train.shape)
        # os_train.shape = [batch_size*out_len, state_num]
        # s_train.shape = [batch_size*out_len]
        loss = F.mse_loss(o_train, y_train) + F.cross_entropy(os_train, s_train)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batch_train += 1
    train_loss = train_loss / n_batch_train

    val_loss, n_batch_val = 0, 0
    with torch.no_grad():
        for batch in val_provider.feed(**val_kwag):
            m.eval()
            x_val, y_val, s_val = batch
            x_val = torch.tensor(x_val, dtype=torch.float, device=device)
            y_val = torch.tensor(y_val, dtype=torch.float, device=device)
            s_val = torch.tensor(s_val, dtype=torch.long, device=device)
            op_val, os_val = m(x_val)
            op_val = torch.reshape(op_val, (op_val.shape[0], out_len, state_num))
            os_val = torch.reshape(os_val, (os_val.shape[0], out_len, state_num))
            oss_val = F.softmax(os_val, dim=-1)
            o_val = torch.sum(oss_val * op_val, dim=-1, keepdim=False)
            os_val = os_val.flatten(0, 1)
            s_val = s_val.flatten(0, 1)
            val_loss += F.mse_loss(o_val, y_val).item() + F.cross_entropy(os_val, s_val).item()
            n_batch_val += 1

    val_loss = val_loss / n_batch_val

    print('>>> Epoch {}: train mse loss {:.6f}, val mse loss {:.6f}'.format(epoch, train_loss, val_loss))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = epoch

        if not os.path.exists('state_dict/'):
            os.mkdir('state_dict/')
        torch.save(m.state_dict(), best_state_dict_path + '.pkl')

    elif best_val_epoch + args.patience < epoch:
        print('>>> Early stopping')
        break

    print('>>> Best model is at epoch {}'.format(best_val_epoch))
    lr_scheduler.step(best_val_loss)

# test
test_len = test_set_x.size - (offset - out_len // 2) * 2
pred = np.zeros((test_len))
pred_p = np.zeros((test_len, state_num))
pred_s = np.zeros((test_len, state_num))
gt = test_set_y[offset - out_len // 2: -offset + out_len // 2]
gt_s = test_set_s[offset - out_len // 2: -offset + out_len // 2]
# m.load_state_dict(torch.load('state_dict/microwave.pkl'))
m.load_state_dict(torch.load(best_state_dict_path + '.pkl'))
m.eval()
datanum = 0
ave = np.ones((test_len)) * out_len
ave[:out_len - 1] = np.arange(1, out_len)
ave[-(out_len - 1):] = np.arange(out_len - 1, 0, -1)
ave_s = np.tile(ave, (state_num, 1))
ave_s = ave_s.T
with torch.no_grad():
    for batch in test_provider.feed(**test_kwag):
        x_test, y_test, s_test = batch
        x_test = torch.tensor(x_test, dtype=torch.float, device=device)
        y_test = torch.tensor(y_test, dtype=torch.float, device=device)
        s_test = torch.tensor(s_test, dtype=torch.int, device=device)
        op_test, os_test = m(x_test)
        op_test = torch.reshape(op_test, (op_test.shape[0], out_len, state_num))
        os_test = torch.reshape(os_test, (os_test.shape[0], out_len, state_num))
        os_test = F.softmax(os_test, dim=-1)
        o_test = torch.sum(os_test * op_test, dim=-1, keepdim=False)

        batch_pred = o_test.cpu().numpy()
        batch_pred_p = op_test.cpu().numpy()
        batch_pred_s = os_test.cpu().numpy()

        for i in range(batch_pred.shape[0]):
            pred[datanum:datanum + out_len] += batch_pred[i]
            pred_p[datanum:datanum + out_len] += batch_pred_p[i]
            pred_s[datanum:datanum + out_len] += batch_pred_s[i]
            datanum += 1

pred, pred_p, pred_s, gt, gt_s = np.vstack(pred / ave), np.vstack(pred_p / ave_s), np.vstack(pred_s / ave_s), np.vstack(
    gt), np.vstack(gt_s)

# max_power = params_appliance[args.appliance_name]['max_on_power']
# threshold = params_appliance[args.appliance_name]['on_power_threshold']

# np.savetxt(save_path+args.appliance_name+'_pred_or.txt',pred.flatten(),fmt='%f',newline='\n')
# np.savetxt(save_path+args.appliance_name+'_pred_p_or.txt',pred_p.flatten(),fmt='%f',newline='\n')
# np.savetxt(save_path+args.appliance_name+'_pred_s_or.txt',pred_s.flatten(),fmt='%f',newline='\n')
# np.savetxt(save_path+args.appliance_name+'_gt_or.txt',gt.flatten(),fmt='%f',newline='\n')

pred = pred * std + mean
pred[pred <= 0.0] = 0.0
pred = pred[132:-132]
gt = gt * std + mean
gt[gt <= 0.0] = 0.0
gt = gt[132:-132]
pred_p = pred_p * std + mean
pred_sh = np.array(np.argmax(pred_s, axis=1))

import metric

sample_second = 6.0  # sample time is 6 seconds
print('MAE:{0}'.format(metric.get_abs_error(gt.flatten(), pred.flatten())))
print('SAE:{0}'.format(metric.get_sae(gt.flatten(), pred.flatten(), sample_second)))
print('SAE_Delta:{}'.format(metric.get_sae_delta(gt.flatten(), pred.flatten(), 1200)))
print(metric.get_sae_delta(gt.flatten(), pred.flatten(), 600))

print(np.mean(gt_s.flatten() == pred_sh.flatten()))
# save the pred to files
# savemains = test_set_x[offset:-offset].flatten()*814+522
savegt = gt.flatten()
savegt_s = gt_s.flatten()
savepred = pred.flatten()
savepred_s = pred_sh.flatten()

np.savetxt(save_path + args.appliance_name + '_pred.txt', savepred, fmt='%f', newline='\n')
np.savetxt(save_path + args.appliance_name + '_gt.txt', savegt, fmt='%f', newline='\n')
# np.savetxt(save_path+args.appliance_name+'_mains.txt',savemains,fmt='%f',newline='\n')
np.savetxt(save_path + args.appliance_name + '_pred_p.txt', pred_p, fmt='%f', newline='\n')
np.savetxt(save_path + args.appliance_name + '_gt_s.txt', savegt_s, fmt='%d', newline='\n')
np.savetxt(save_path + args.appliance_name + '_pred_s.txt', savepred_s, fmt='%d', newline='\n')
np.savetxt(save_path + args.appliance_name + '_pred_sp.txt', pred_s, fmt='%f', newline='\n')
