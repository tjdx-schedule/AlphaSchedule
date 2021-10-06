# -*- coding: utf-8 -*-
import numpy as np

from .EnvirConf import envConfig as ec
from .EnvirConf import obConfig

e = 1e-10
class Observation:
    def __init__(self,scheduler):
        self.scheduler = scheduler
        
        self.partNum = ec.partNum
        self.machNum = ec.machNum
        
        self.feaWidth = obConfig.width
        self.feaHeight = obConfig.height
        self.feaArea = self.feaWidth * self.feaHeight
        
        self.feaNum = obConfig.feaNum
        self.stackN = obConfig.stackFrameNum
        
    
    def getState(self):
        obs = []
        
        partFea,machTime = self.partEncode()
        
        obs = self.stackFeature(partFea,machTime)
        return obs

    def partEncode(self):
        e = 1e-10
        cur = np.min(self.scheduler.mach)
        
        hours, deadline, priority = np.hsplit(self.scheduler.part,3)
        
        deadline_cur = (deadline - cur)*(deadline > 0)
        deadline_cur_hours = deadline_cur - hours

        
        slack = -(deadline_cur_hours)/(hours+e)

        partFeatureMat = np.concatenate((hours, priority,
                        deadline_cur,deadline_cur_hours,
                        slack),axis=1)       
        partFeature = partFeatureMat
        machTime = self.scheduler.mach - cur
        
        # feature = np.hstack((partFeature,machTime))
        return [partFeature,machTime]
    
    def stackFeature(self,partFea,machTime):
        part_obs = partFea.T.reshape(-1,self.feaWidth,self.feaHeight)
        
        mach_arr = np.zeros(self.feaArea)
        mach_arr[:self.machNum] =  machTime
        mach_obs = mach_arr.reshape(-1,self.feaWidth,self.feaHeight)
        
        obs = np.concatenate((part_obs,mach_obs),axis=0)
        return obs