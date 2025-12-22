#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from .vec_env import VecEnv
from .util import copy_obs_dict, dict_to_obs, obs_space_info

class DummyMy(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space, env.observation_space_value)
        obs_space = env.observation_space
        value_obs_space = env.observation_space_value
        
#        self.value_observation_space = value_obs_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        
        act_num = [i.n for i in env.action_space]
        self.buf_value_states = np.zeros((self.num_envs,*value_obs_space.shape),dtype=np.float32)
        self.buf_action_masks = []
        
        index_arr = [[act_num[0]],
                     [act_num[0],act_num[1]]]
        for i in range(len(act_num)):
            if i > 1:
                index_arr.append([act_num[1],act_num[i]])
            init_mat = np.zeros((self.num_envs,*index_arr[i]),dtype=np.float32)
            self.buf_action_masks.append(init_mat)
        # self.buf_stock_masks = np.zeros((self.num_envs,act_num[0]),dtype=np.float32)
        # self.buf_mack_masks = np.zeros((self.num_envs,act_num[0],act_num[1]),dtype=np.float32)
        # self.buf_tool_masks = np.zeros((self.num_envs,act_num[1],act_num[2]),dtype=np.float32)
        # self.buf_worker_masks = np.zeros((self.num_envs,act_num[1],act_num[3]),dtype=np.float32)
               
        self.actions = None
        self.spec = self.envs[0].spec

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            # if isinstance(self.envs[e].action_space, spaces.Discrete):
            #    action = int(action)

            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)
        
    def get_action_mask(self):
        for e in range(self.num_envs):
            self.buf_value_states[e], *e_action_masks = self.envs[e].getActionMask()
            for i in range(len(e_action_masks)):
                self.buf_action_masks[i][e] = e_action_masks[i]
        action_masks = [np.copy(self.buf_value_states)]
        for i in range(len(e_action_masks)):
            action_masks.append(np.copy(self.buf_action_masks[i]))
        return action_masks
            