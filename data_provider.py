# -*- coding: utf-8 -*-

import numpy as np

class S2P_Slider(object):
    def __init__(self, batch_size, shuffle, offset, length):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.offset = offset
        self.length = length

    def feed(self, inputs, targets):
        total_size = inputs.shape[0] - self.length + 1
        if self.batch_size < 0:
            self.batch_size = total_size

        indices = np.arange(total_size)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, total_size, self.batch_size):
            excerpt = indices[start_idx:start_idx + self.batch_size]
            yield np.array([inputs[idx:idx + self.length] for idx in excerpt]), \
                    targets[excerpt + self.offset].reshape(-1, 1)

class S2S_Slider(object):

    def __init__(self, batch_size, shuffle, offset, length, out_len):

        self.batchsize = batch_size
        self.shuffle = shuffle
        self.offset = offset
        self.length = length
        self.out_len = out_len

    def feed(self, inputs, targets):

        inputs, targets = inputs.flatten(), targets.flatten()
        assert inputs.size == targets.size

        max_batchsize = inputs.size - self.length
        if self.batchsize < 0:
            self.batchsize = max_batchsize

        indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            
            yield np.array([inputs[idx:idx + self.length] for idx in excerpt]), \
                  np.array([targets[idx+self.offset-(self.out_len//2) : idx+self.offset+(self.out_len-self.out_len//2)] for idx in excerpt])

class S2P_State_Slider(object):
    def __init__(self, batch_size, shuffle, offset, length):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.offset = offset
        self.length = length

    def feed(self, inputs, targets, targets_s):
        total_size = inputs.shape[0] - self.length + 1
        if self.batch_size < 0:
            self.batch_size = total_size

        indices = np.arange(total_size)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, total_size, self.batch_size):
            excerpt = indices[start_idx:start_idx + self.batch_size]
            yield np.array([inputs[idx:idx + self.length] for idx in excerpt]), \
                    targets[excerpt + self.offset].reshape(-1, 1), targets_s[excerpt + self.offset].reshape(-1, 1)

class S2P_State_Slider_drop(object):
    def __init__(self, batch_size, shuffle, offset, length, threshold):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.offset = offset
        self.length = length
        self.threshold = threshold

    def feed(self, inputs, targets, targets_s):
        total_size = inputs.shape[0] - self.length + 1
        
        indices = []
        indices_0 = []
        for start_idx in range(0, total_size):
            '''
            if start_idx%2 == 0:
                indices.append(start_idx)
                continue
             '''   
            flag = 0
            for i in targets[start_idx:start_idx + 400]:
                if i > self.threshold:
                    indices.append(start_idx)
                    flag = 1
                    break
            if flag == 0:
                indices_0.append(start_idx)
                   
        for i in range(0,len(indices_0)):
            if i%2 == 0:
                indices.append(indices_0[i])
                
        indices = np.asarray(indices)
        total_size = len(indices)
        
        if self.batch_size < 0:
            self.batch_size = total_size

        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, total_size, self.batch_size):
            excerpt = indices[start_idx:start_idx + self.batch_size]
            yield np.array([inputs[idx:idx + self.length] for idx in excerpt]), \
                    targets[excerpt + self.offset].reshape(-1, 1), targets_s[excerpt + self.offset].reshape(-1, 1)            
            
class S2S_State_Slider(object):

    def __init__(self, batch_size, shuffle, offset, length, out_len):

        self.batchsize = batch_size
        self.shuffle = shuffle
        self.offset = offset
        self.length = length
        self.out_len = out_len

    def feed(self, inputs, targets, targets_s):

        inputs, targets = inputs.flatten(), targets.flatten()
        assert inputs.size == targets.size

        max_batchsize = inputs.size - self.length + 1
        if self.batchsize < 0:
            self.batchsize = max_batchsize

        indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            
            yield np.array([inputs[idx:idx + self.length] for idx in excerpt]), \
                  np.array([targets[idx+self.offset-(self.out_len//2) : idx+self.offset+(self.out_len-self.out_len//2)] for idx in excerpt]),\
                  np.array([targets_s[idx+self.offset-(self.out_len//2) : idx+self.offset+(self.out_len-self.out_len//2)] for idx in excerpt])

class S2S_State_Slider_drop_winlen(object):

    def __init__(self, batch_size, shuffle, offset, length, out_len, threshold):

        self.batchsize = batch_size
        self.shuffle = shuffle
        self.offset = offset
        self.length = length
        self.out_len = out_len
        self.threshold = threshold

    def feed(self, inputs, targets, targets_s):

        inputs, targets = inputs.flatten(), targets.flatten()
        assert inputs.size == targets.size

        total_size = inputs.size - self.length + 1
        indices = []
        indices_0 = []
        for start_idx in range(0, total_size):
            '''
            if start_idx%2 == 0:
                indices.append(start_idx)
                continue
             '''   
            flag = 0
            for i in targets[start_idx:start_idx + self.length]:
                if i > self.threshold:
                    indices.append(start_idx)
                    flag = 1
                    break
            if flag == 0:
                indices_0.append(start_idx)
                   
        for i in range(0,len(indices_0)):
            if i%2 == 0:
                indices.append(indices_0[i])

        max_batchsize = len(indices)
        if self.batchsize < 0:
            self.batchsize = max_batchsize

        #indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            
            yield np.array([inputs[idx:idx + self.length] for idx in excerpt]), \
                  np.array([targets[idx+self.offset-(self.out_len//2) : idx+self.offset+(self.out_len-self.out_len//2)] for idx in excerpt]),\
                  np.array([targets_s[idx+self.offset-(self.out_len//2) : idx+self.offset+(self.out_len-self.out_len//2)] for idx in excerpt])

class S2S_State_Slider_drop_crf(object):

    def __init__(self, batch_size, shuffle, offset, length, out_len, threshold):

        self.batchsize = batch_size
        self.shuffle = shuffle
        self.offset = offset
        self.length = length
        self.out_len = out_len
        self.threshold = threshold

    def feed(self, inputs, targets, targets_s):

        inputs, targets = inputs.flatten(), targets.flatten()
        assert inputs.size == targets.size

        total_size = inputs.size - self.length + 1
        indices = []
        indices_0 = []
        for start_idx in range(0, total_size):
            '''
            if start_idx%2 == 0:
                indices.append(start_idx)
                continue
             '''   
            flag = 0
            for i in targets[start_idx+self.offset-(self.out_len//2) : start_idx+self.offset+(self.out_len-self.out_len//2)]:
                if i != targets[start_idx+self.offset-(self.out_len//2)]:
                    indices.append(start_idx)
                    flag = 1
                    break
            if flag == 0:
                indices_0.append(start_idx)
        '''           
        for i in range(0,len(indices_0)):
            if i%2 == 0:
                indices.append(indices_0[i])
        '''

        max_batchsize = len(indices)
        if self.batchsize < 0:
            self.batchsize = max_batchsize

        #indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            
            yield np.array([inputs[idx:idx + self.length] for idx in excerpt]), \
                  np.array([targets[idx+self.offset-(self.out_len//2) : idx+self.offset+(self.out_len-self.out_len//2)] for idx in excerpt]),\
                  np.array([targets_s[idx+self.offset-(self.out_len//2) : idx+self.offset+(self.out_len-self.out_len//2)] for idx in excerpt])
            
class S2S_State_Slider_non_overlapping(object):

    def __init__(self, batch_size, shuffle, offset, length, out_len):

        self.batchsize = batch_size
        self.shuffle = shuffle
        self.offset = offset
        self.length = length
        self.out_len = out_len

    def feed(self, inputs, targets, targets_s):

        inputs, targets = inputs.flatten(), targets.flatten()
        assert inputs.size == targets.size

        total_size = inputs.size - self.length + 1
        indices = []
        for start_idx in range(0, total_size):
            if start_idx%self.out_len == 0:
                indices.append(start_idx)
            
        max_batchsize = len(indices)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            
            yield np.array([inputs[idx:idx + self.length] for idx in excerpt]), \
                  np.array([targets[idx+self.offset-(self.out_len//2) : idx+self.offset+(self.out_len-self.out_len//2)] for idx in excerpt]),\
                  np.array([targets_s[idx+self.offset-(self.out_len//2) : idx+self.offset+(self.out_len-self.out_len//2)] for idx in excerpt])