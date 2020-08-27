# -*- coding: utf-8 -*-
import gym
import numpy as np

from .scheduler import Scheduler
from .obs import Observation

from .EnvirConf import envConfig as ec
from .EnvirConf import obConfig as oc

class ScheEnv(gym.Env):
    def __init__(self, seed = 0, mode=None):
        self.envConfig = ec
        self.obsConfig = oc
        
        self.partNum = self.envConfig.partNum
        self.machNum = self.envConfig.machNum
        
        self.sche = Scheduler(mode,seed)
        self.obsGen = Observation(self.sche)
        self.actObsInit()
        
    def reset(self):
        self.sche.reset()
        canvas_obs = self.obsGen.getState()
        return canvas_obs
    
    def step(self, action):
        reward = self.sche.step(action[0])#mult-actions frame
        other_reward , done , grade =  self.sche.is_end()
       
        canvas_obs = self.obsGen.getState()
        reward += other_reward
        info = {}
        if done:
            info['episode'] = {'r':round(grade)}
        return canvas_obs,reward,done,info
        
    def getActionMask(self):
        vObs = np.array([0],dtype=np.float)
        partAvai = self.sche.available()
        
        partMask = np.zeros(self.partNum)
        partMask[partAvai] = 1
        return vObs,partMask
    
    def actObsInit(self):
        self.action_space = [
                gym.spaces.Discrete(self.partNum),
                ]
        
        self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.obsGen.feaNum,self.obsGen.feaWidth,\
                       self.obsGen.feaHeight),
                dtype=np.float32,
                )
        
        self.observation_space_value  = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32,
                )



if __name__ == '__main__':
    value = 0
    stateLi = []
    vLi = []
    a = ScheEnv(0,'train')
    state = a.reset()
    done = False
    while not done:
        v_obs, avai = a.getActionMask()
        act = np.random.choice(avai)
        state, reward, done, info = a.step(act)
        stateLi.append(state)
        vLi.append(v_obs)
        value += reward
    print(info['episode']['r'])
    print(value)
