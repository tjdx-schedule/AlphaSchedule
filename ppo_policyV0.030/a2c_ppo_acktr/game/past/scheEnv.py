# -*- coding: utf-8 -*-
import gym
import numpy as np

from .scheduler import SkipDynamicScheduler,ranChoose,machRanChoose
from .obs import Observation

from .EnvirConf import envConfig as ec
from .EnvirConf import obConfig as oc

class ScheEnv(gym.Env):
    def __init__(self, seed = 0, mode=None):
        self.stock = ec.STOCKNUM
        self.machNum = ec.MACHNUM
        
        self.sche = SkipDynamicScheduler(seed)
        self.obsGen = Observation(self.sche)
        self.actObsInit()
        
    def reset(self):
        self.sche.reset()
        canvas_obs = self.obsGen.drawConState()
        return canvas_obs
    
    def step(self, action):
        reward,done,info = self.sche.step(action)
        canvas_obs = self.obsGen.drawConState()
        return canvas_obs,reward,done,info
        
    def getActionMask(self):
        vObs = self.obsGen.valueState()
        stockMask,machMask = self.sche.getMask()
        return vObs,stockMask,machMask
    
    def actObsInit(self):
        self.action_space = [
                gym.spaces.Discrete(self.stock),
                gym.spaces.Discrete(self.machNum),
                ]
        
        self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.obsGen.colConLength,self.obsGen.feaWidth,\
                       self.obsGen.feaHeight),
                dtype=np.float32,
                )
        
        self.observation_space_value  = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(oc.feaNum,oc.width,oc.height),
                dtype=np.float32,
                )



if __name__ == '__main__':
    value = 0
    stateLi = []
    vLi = []
    a = ScheEnv(0)
    state, reward, done, info = a.reset()
    while not done:
        v_obs, partMask, machMask = a.getActionMask()
        act = ranChoose(partMask)
        machAct = machRanChoose(machMask,act)
        
        state, reward, done, info = a.step([act,machAct])
        stateLi.append(state)
        vLi.append(v_obs)
        value += reward
    print(info['episode']['r'])
    print(value/600)
