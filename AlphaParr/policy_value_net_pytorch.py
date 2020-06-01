# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from config import scheConfig

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, partNum,machNum):
        super(Net, self).__init__()
        input_size = 9*partNum+machNum
        # common layers        
        self.fc0 = nn.Linear(input_size,512)
        self.fc1 = nn.Linear(512,512)
        # self.fc2 = nn.Linear(1024,512)
        
        # action policy layers
        self.act_fc0 = nn.Linear(512,256)
        self.act_fc1 = nn.Linear(256,partNum)
        # state value layers
        self.val_fc0 = nn.Linear(512,256)
        self.val_fc1 = nn.Linear(256, 1)

    def forward(self, state_input):
        state_p,state_m = state_input
        # common layers
        x = state_p       
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        # action policy layers
        x_act = F.relu(self.act_fc0(x))
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_fc0(x))
        x_val = F.tanh(self.val_fc1(x_val))
        # x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.l2_const = 1e-4  # coef of l2 penalty
        p, o = scheConfig.partNum,scheConfig.machNum
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(p, o).cuda()
        else:
            self.policy_value_net = Net(p, o)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)
        
        self.save_dict = self.load_model(model_file)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_first_batch = np.array(state_batch[0])
        state_second_batch = np.array(state_batch[1]).reshape(-1,1)
        if self.use_gpu:
            state_first_batch_var = Variable(torch.FloatTensor(state_first_batch).cuda())
            state_second_batch_var = Variable(torch.FloatTensor(state_second_batch).cuda())
            log_act_probs, value = self.policy_value_net([state_first_batch_var,state_second_batch_var])
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            
            value_add = value.data.cpu().numpy()
        else:
            state_first_batch_var = Variable(torch.FloatTensor(state_first_batch))
            state_second_batch_var = Variable(torch.FloatTensor(state_second_batch))
            log_act_probs, value = self.policy_value_net([state_first_batch_var,state_second_batch_var])
            act_probs = np.exp(log_act_probs.data.numpy())
            
            value_add = value.data.numpy() 
        # value_add = value_pre + state_batch[1]
        return act_probs, value_add+state_second_batch

    def policy_value_fn(self, state, legal_positions):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        # legal_positions = board.availables
        state_first, state_second = state
        current_state_first = np.ascontiguousarray(np.expand_dims(state_first,0))
        current_state_second = np.ascontiguousarray(np.expand_dims(state_second,0))

        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    [Variable(torch.from_numpy(current_state_first)).cuda().float(),
                     Variable(torch.from_numpy(current_state_second)).cuda().float()])
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    [Variable(torch.from_numpy(current_state_first)).float(),
                     Variable(torch.from_numpy(current_state_second)).float()])
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = act_probs[legal_positions]/np.sum(act_probs[legal_positions])
        act_probs = zip(legal_positions, act_probs)
        # act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0] + state_second
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        state_first_batch = np.array(state_batch[0])
        state_second_batch = np.array(state_batch[1])
        winner_batch = winner_batch - state_second_batch
        if self.use_gpu:
            state_first_batch = Variable(torch.FloatTensor(state_first_batch).cuda())
            state_second_batch = Variable(torch.FloatTensor(state_second_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_first_batch = Variable(torch.FloatTensor(state_first_batch))
            state_second_batch = Variable(torch.FloatTensor(state_second_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        # set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net([state_first_batch,state_second_batch])
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        # return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        return value_loss.item(), policy_loss.item(), \
            loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file, best_ratio=0.0, vec_norm = None, logger=None):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        save_dict = {'net_params':net_params,'best_ratio':best_ratio,
                     'vec_norm': vec_norm,
                     'logger': logger}
        torch.save(save_dict, model_file)
    
    def load_model(self, model_file):
        win_ratio = None
        vec_norm = None
        logger = None
        if model_file:
            state_dict = torch.load(model_file)
            net_params = state_dict['net_params']
            self.policy_value_net.load_state_dict(net_params)
            win_ratio = state_dict['best_ratio']
            vec_norm = state_dict['vec_norm']
            logger = state_dict['logger']
        return [win_ratio,vec_norm,logger]
        # else:
        #     raise Exception('model Path is error!')
    
    def update_model(self, net_params):
        self.policy_value_net.load_state_dict(net_params)