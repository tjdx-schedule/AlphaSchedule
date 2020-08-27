#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .baselines_com.vec_env.shmem_vec_my import ShmemMy
# from .baselines_com.vec_env.dummy_vec_my import DummyMy

from .scheEnv import ScheEnv

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
    return envs



