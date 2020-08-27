import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Categorical
from a2c_ppo_acktr.game.EnvirConf import obConfig

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, v_obs_shape=None, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if v_obs_shape is not None:
                base = ResNet
            else:
                raise NotImplementedError
                
        input_size = obConfig.width * obConfig.height

        self.base = base(obs_shape[0], v_obs_shape[0], \
                         input_size, **base_kwargs)
        
        self.act_n = len(action_space)
        self.act_num = []
        self.act_index = [0]
        
        dists = []
        if len(action_space) > 0:
            for act_sp in action_space:
                num_outputs = act_sp.n                
                dists.append(Categorical(self.base.policy_output_size, num_outputs))
                
                self.act_num.append(num_outputs)
                self.act_index.append(self.act_index[-1]+num_outputs)
            self.dists = nn.ModuleList(dists)
        else:
            raise NotImplementedError

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, vec_inputs, rnn_hxs, masks, action_masks, deterministic=False):
        self.base.eval()
        
        value, actor_features, rnn_hxs = self.base(inputs, vec_inputs, rnn_hxs, masks)
        
        device = value.device
        size_zeros = value.size(0)
        
        action_log_probs = torch.zeros((size_zeros,1)).to(device)
        action_multi = torch.zeros((size_zeros,self.act_n),dtype=torch.long).to(device)
        
        squeeze_act = None
        for i in range(self.act_n):
            if i == 0:
                part_action_mask = action_masks[i]
            elif i == 1:
                part_action_mask = action_masks[i][range(size_zeros),squeeze_act,:]
            elif i == 2:
                mach_squeeze_act = squeeze_act
                part_action_mask = action_masks[i][range(size_zeros),mach_squeeze_act,:]
            elif i == 3:
                part_action_mask = action_masks[i][range(size_zeros),mach_squeeze_act,:]
            else:
                raise NotImplementedError
                
            dist = self.dists[i](actor_features, part_action_mask)

            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
                
            squeeze_act = action.squeeze(-1)       
            action_multi[:,i] = squeeze_act
            action_log_probs += dist.log_probs(action)
        return value, action_multi, action_log_probs, rnn_hxs

    def get_value(self, inputs, vec_inputs, rnn_hxs, masks):
        self.base.eval()
        
        value, _, _ = self.base(inputs, vec_inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, vec_inputs, rnn_hxs, masks, action, action_masks):
        self.base.train()
        
        value, actor_features, rnn_hxs = self.base(inputs, vec_inputs, rnn_hxs, masks)
#        device = value.device
        size_zero = value.size(0)
        
        list_action_log_probs = []
        list_dist_entropy = []
        action_multi = action
        
        for i in range(self.act_n):
            if i == 0:
                part_action_mask = action_masks[i]
            elif i == 1:
                part_action_mask = action_masks[i][range(size_zero),action_multi[:,0],:]
            elif i == 2:
                part_action_mask = action_masks[i][range(size_zero),action_multi[:,1],:]
            elif i == 3:
                part_action_mask = action_masks[i][range(size_zero),action_multi[:,1],:]
            else:
                raise NotImplementedError
                
            dist = self.dists[i](actor_features, part_action_mask)
            
            list_action_log_probs.append(dist.log_probs(action_multi[:,i].unsqueeze(-1)))
            list_dist_entropy.append(dist.entropy().mean())
            
        sum_action_log_probs = sum(list_action_log_probs)
        sum_dist_entropy = sum(list_dist_entropy)
        return value, sum_action_log_probs, sum_dist_entropy, rnn_hxs
    
    def save_weight(self,path,ret_rms):
        path += '/weight.model'
        
        weight_dict = {}
        weight_dict['com_base'] = self.base.res_tower.state_dict()
        weight_dict['policy_head'] = self.base.policy_head.state_dict()
        weight_dict['dist0_fc'] = self.dists[0].linear.state_dict()
        weight_dict['value_head'] = self.base.value_head.state_dict()
        
        weight_dict['ret_rms_mean'] = ret_rms.mean
        weight_dict['ret_rms_var'] = ret_rms.var
        
        torch.save(weight_dict,path)

class ResNet(nn.Module):
    def __init__(self, num_inputs_canvas, num_inputs_value, input_size):
        super().__init__()
        filter_num = 128
        resblock_num = 9
        num_inputs = num_inputs_canvas
        
        self.policy_output_size = input_size * 2
        self.recurrent_hidden_state_size = 1
        
        self.res_tower = ResidualTower(resblock_num,filter_num,num_inputs)
        # action policy layers
        self.policy_head = PolicyHead(input_size,filter_num)
        # state value layers
        self.value_head = ValueHead(input_size,filter_num)
        
        self.train()
        
    def forward(self, visual_inputs, vec_inputs, rnn_hxs, masks):
        x = visual_inputs
        
        x = self.res_tower(x)       
        # action policy layers
        x_act = self.policy_head(x)
        # state value layers
        value = self.value_head(x)
        return value, x_act, rnn_hxs



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
        # self.act_fc0 = nn.Linear(2*input_size,input_size)

    def forward(self, x):
        x = self.act_conv(x)
        x = self.act_bn(x)
        x =  F.relu(x)
        x = x.view(-1,2*self.input_size)
        
        x_act = x
        # x = self.act_fc0(x)
        return x_act
    
        
        