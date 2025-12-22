#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
import torch
from gym.spaces.box import Box

# from .baselines_com.atari_wrappers import WarpFrame
from .baselines_com.vec_env.vec_env import VecEnvWrapper
from .baselines_com.vec_env.shmem_vec_my import ShmemMy
from .baselines_com.vec_env.dummy_vec_my import DummyMy

from .scheEnv import ScheEnv

def make_env(env_id, seed, rank, log_dir, env_mode):#baseline package
    def _thunk():
        env = ScheEnv(seed=rank, mode = env_mode)        
        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  env_mode,
                  num_frame_stack=None,
                  seed_interval = 1):
    envs = [
        make_env(env_name, seed, i*seed_interval, log_dir, env_mode)
        for i in range(num_processes)
    ]    
    
    if len(envs) > 1:
        envs = ShmemMy(envs, context='fork')
    else:
        envs = DummyMy(envs)

    if gamma is None:
        envs = VecNormalize(envs, ret=False)
    else:
        envs = VecNormalize(envs, gamma=gamma)
   
    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 1, device)

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


class VecNormalize(VecEnvWrapper):
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, use_tf=False):
        VecEnvWrapper.__init__(self, venv)
        self.training = True
        
        from .baselines_com.vec_env.running_mean_std import RunningMeanStd
        self.ob_rms = RunningMeanStd(shape=self.value_observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
    
    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        if self.ret_rms:
            if self.training:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos
        
    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return obs
    
    def get_action_mask(self):
        action_masks = self.venv.get_action_mask()
        # action_masks[0] = self._obfilt(action_masks[0])
        return action_masks
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()

        
