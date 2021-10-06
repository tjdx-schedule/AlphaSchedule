#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from envs import make_vec_envs

envs = make_vec_envs(0,'test',10)
envs.init_board()

k = envs.reset()
print('---')
for z in k:
    print(z.sum())
a,b = envs.get_action_mask()
# envs.step([1]*10)