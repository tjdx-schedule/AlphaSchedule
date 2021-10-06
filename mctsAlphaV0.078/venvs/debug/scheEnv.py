# -*- coding: utf-8 -*-
import gym
import numpy as np

from scheduler import Scheduler
from obs import Observation

from EnvirConf import envConfig as ec
from EnvirConf import obConfig as oc

class ScheEnv(gym.Env):
    def __init__(self, seed, mode):
        self.envConfig = ec
        self.obsConfig = oc
        
        self.partNum = self.envConfig.partNum
        self.machNum = self.envConfig.machNum
        
        self.sche = Scheduler(mode,seed)
        self.obsGen = Observation(self.sche)
        self.actObsInit()

    def init_board(self):
        self.sche.reset(update_example=True)
        self.availables = self.sche.available()
        self.canvas_obs = self.obsGen.getState()
        
    def reset(self):
        self.sche.reset(update_example=False)
        self.canvas_obs = self.obsGen.getState()
        self.availables = self.sche.available()
        return self.canvas_obs
    
    def step(self, action):
        action = int(action)
        if action in self.availables:
            self.sche.step(action)#mult-actions frame
            self.availables = self.sche.available()
            
        done , grade =  self.sche.is_end()
        self.canvas_obs = self.obsGen.getState()
        info = {}
        if done:
            info['episode'] = {'r':round(grade)}
        return self.canvas_obs,0,done,info
        
    def getActionMask(self):
        vObs = np.array([0],dtype=np.float)
        partAvai = self.availables
        
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
    
    def node_board(self,treeNode):
        actSeq = treeNode.act_seq
        self.reset()
        for act in actSeq:
            self.step(act)
        return self
    
    def game_end(self):#undo:grade norm
        done, grade = self.sche.is_end()
        return done,grade

if __name__ == '__main__':
    value = 0
    stateLi = []
    vLi = []
    a = ScheEnv(0,'test')
    a.init_board()
    state_0 = a.reset()
    done = False
    while not done:
        v_obs, avai = a.getActionMask()
        act = np.random.choice(a.availables)
        print(act)
        state, reward, done, info = a.step(act)
        stateLi.append(state)
        vLi.append(v_obs)
        value += reward
    print(info['episode']['r'])
    print(value)
