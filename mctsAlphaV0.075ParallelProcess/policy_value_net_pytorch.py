# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from venvs.EnvirConf import envConfig ,obConfig

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, p,m,n,f,w,h):
        super(Net, self).__init__()
        filter_num = 128
        input_size = w*h
        resblock_num = 9
        num_inputs = n*f
        
        # common layers        
        self.res_tower = ResidualTower(resblock_num,filter_num,num_inputs)
        # action policy layers
        self.policy_head = PolicyHead(input_size,filter_num)
        self.dist0_linear = nn.Linear(2*input_size,p)
        
        # state value layers
        self.value_head = ValueHead(input_size,filter_num)
        
        
    def forward(self, state_input):
        x, action_masks = state_input
        # common layers
        x = self.res_tower(x) 
        
        # action policy layers
        x_act = self.policy_head(x)
        x_act = self.dist0_linear(x_act)+(action_masks+1e-45).log()
        x_act = F.log_softmax(x_act)
        
        # state value layers
        x_val = self.value_head(x)
        return x_act, x_val


class PolicyValueNet:
    """policy-value network """
    def __init__(self, model_file=None, use_gpu=False, is_train = True):
        self.use_gpu = use_gpu
        self.is_train = is_train
        self.lr = 1e-4
        self.l2_const = 1e-4  # coef of l2 penalty
        
        self.p, self.m = envConfig.partNum,envConfig.machNum
        self.n, self.f = obConfig.stackFrameNum,obConfig.feaNum
        self.w, self.h = obConfig.width, obConfig.height
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(self.p, self.m, self.n, self.f, self.w, self.h).cuda()
        else:
            self.policy_value_net = Net(self.p, self.m, self.n, self.f, self.w, self.h)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    lr=self.lr,weight_decay=self.l2_const) \
            if self.is_train else None
        self.iter = 1 
        
        if self.is_train:
            self.policy_value_net.train()
        else:
            self.policy_value_net.eval()
                    
        self.save_dict = self.load_model_mpi(model_file)

    def policy_value_sess(self, state_batch, action_masks):
        current_state_first = state_batch 
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    [torch.from_numpy(current_state_first).cuda().float(),
                     torch.from_numpy(action_masks).cuda().float()])
            log_act_probs = log_act_probs.data.cpu()
            value = value.data.cpu()
        else:
            log_act_probs, value = self.policy_value_net(
                    [torch.from_numpy(current_state_first).float(),
                     torch.from_numpy(action_masks).float()])
            log_act_probs = log_act_probs.data
            value = value.data
        return log_act_probs, value

    def policy_value_act(self, state_batch, action_mask):
        log_act_probs, _ = self.policy_value_sess(state_batch,action_mask)
        acts = log_act_probs.argmax(dim=-1).numpy()
        return acts
        
    def policy_value_fn(self, state, action_mask):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        current_state_first = np.ascontiguousarray(np.expand_dims(state,0))
        current_action_mask = np.ascontiguousarray(np.expand_dims(action_mask,0))
        
        log_act_probs,value = self.policy_value_sess(current_state_first,
                                                     current_action_mask)
        act_probs = np.exp(log_act_probs.numpy().flatten())
        value = value.numpy()[0][0]
        return act_probs, value
       
    def load_pretrained_weight(self, path, load_way='ori'):
        map_way = self._load_way_setting(load_way)
        weight_dict = torch.load(path,map_location=map_way)
        
        self.policy_value_net.res_tower.load_state_dict(weight_dict['com_base'])
        self.policy_value_net.policy_head.load_state_dict(weight_dict['policy_head'])
        self.policy_value_net.dist0_linear.load_state_dict(weight_dict['dist0_fc'])
        self.policy_value_net.value_head.load_state_dict(weight_dict['value_head'])
        

    def restore_net_param(self, model_path, load_way='ori'):
        map_way = self._load_way_setting(load_way)
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path,map_location=map_way)
            except:
                time.sleep(5)
                state_dict = torch.load(model_path,map_location=map_way)
            net_params = state_dict['net_params']
            self.policy_value_net.load_state_dict(net_params)
    
    def _load_way_setting(self,load_way):
        if load_way == 'g2c':
            map_way = lambda storage, loc: storage
        elif load_way == 'ori':
            map_way = None
        elif load_way == 'c2g':
            map_way = lambda storage, loc: storage.cuda(0)
        else:
            raise Exception('load state_diect way error!')
        return map_way
    
    def load_model_mpi(self, model_file, load_way ='ori'):
        vec_norm = None
        if model_file:
            self.restore_net_param(model_file+self.current_net_name, load_way)
            
            state_dict = torch.load(model_file+self.current_other_name)
            optimizer_param = state_dict['optimizer']
    
            if self.is_train and optimizer_param is not None:
                self.optimizer.load_state_dict(optimizer_param)
            self.iter = state_dict['iter']     
            vec_norm = state_dict['vec_norm']
        return vec_norm
    
    
#***********************Module**************************************    
def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )
   
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, n, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(n, num_channels, kernel_size=3,stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        
    def forward(self, state):
        x = state
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x

#***********************component**************************************
class ResidualTower(torch.nn.Module):
    def __init__(self,resblock_num,filter_num,num_inputs):
        super().__init__()
        self.pre_bn = torch.nn.BatchNorm2d(num_inputs)
        
        self.convLayer = ConvLayer(num_inputs,filter_num)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(filter_num) for _ in range(resblock_num)])
        
    def forward(self, visual_inputs):
        x = self.pre_bn(visual_inputs)
        
        x = self.convLayer(x)
        for block in self.resblocks:
            x = block(x)
        return x

class ValueHead(torch.nn.Module):
    def __init__(self, input_size, num_channels):
        super().__init__()
        self.input_size = input_size
        self.val_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.val_bn = torch.nn.BatchNorm2d(1)
        self.val_fc0 = nn.Linear(input_size,num_channels//2)
        self.val_fc1 = nn.Linear(num_channels//2,1)
    
    def forward(self, x):
        x = self.val_conv(x)
        x = self.val_bn(x)
        x =  F.relu(x)
        
        x = x.view(-1,self.input_size)
        
        x = self.val_fc0(x)
        x = F.relu(x)
        x = self.val_fc1(x)
        return x

class PolicyHead(torch.nn.Module):
    def __init__(self, input_size, num_channels):  
        super().__init__()
        self.input_size = input_size
        self.act_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.act_bn = torch.nn.BatchNorm2d(2)

    def forward(self, x):
        x = self.act_conv(x)
        x = self.act_bn(x)
        x =  F.relu(x)
        x = x.view(-1,2*self.input_size)
        
        return x

