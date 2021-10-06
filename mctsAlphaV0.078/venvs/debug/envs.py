#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from baselines_com.vec_env.shmem_vec_my import ShmemMy
from baselines_com.vec_env.vec_env import VecEnvWrapper
# from .baselines_com.vec_env.dummy_vec_my import DummyMy

from scheEnv import ScheEnv

def make_env(rank, env_mode):#baseline package
    def _thunk():
        env = ScheEnv(seed=rank, mode = env_mode)        
        return env

    return _thunk


def make_vec_envs(seed,
                  env_mode,
                  num_processes):
    envs = [
        make_env(seed, env_mode)
        for i in range(num_processes)
    ]    
    
    if len(envs) > 1:
        envs = ShmemMy(envs, context='fork')
    else:
        raise Exception('env num smaller than 2')
    
    envs = VecPyTorch(envs, 'cpu')
    
    return envs



class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
#        if isinstance(actions, torch.LongTensor):
#            # Squeeze the dimension for discrete actions
#            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
    
    def get_action_mask(self):
        action_masks = self.venv.get_action_mask()
        action_masks = list(action_masks)
        for i in range(len(action_masks)):
            action_masks[i] = torch.from_numpy(action_masks[i]).float().to(self.device)
        return action_masks
