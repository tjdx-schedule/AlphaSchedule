# -*- coding: utf-8 -*-
import numpy as np
from .EnvirConf import envConfig as ec
from .EnvirConf import obConfig

e = 1e-10
class Observation:
    def __init__(self,scheduler):
        self.sche = scheduler
        
        self.stock = ec.STOCKNUM
        self.machNum = ec.MACHNUM
        
        self.feaWidth = obConfig.width
        self.feaHeight = obConfig.height
        
        self.drawConsInits()
    
    def drawConsInits(self): #undo:revise 1
        self.dyColorCoef = 0.5
    
        self.rowCon = self.stock
        self.colCon = self.machNum
        # assert (self.rowCon <= 84 and self.colCon <= 84)
        
        self.rowConLength = self.rowCon #
        self.colConLength = self.colCon #


    def drawConState(self):
        # drawing = np.zeros((self.rowConLength,self.colConLength,3),dtype = np.uint8) 
        
        dyEquCons = self.getDyEquCons().T
        drawing = dyEquCons.reshape(self.colConLength,self.feaWidth, self.feaHeight)
        return drawing
    
    def getDyEquCons(self):   
        dyEquCons = self.sche.partEquCons.copy()
        dyEquCons[:,self.sche.machPrio != 0] = self.dyColorCoef
        return dyEquCons
    
    def valueState(self):
        partFeatureMat = self.stockState()
        machFeatureMat = self.machState()
        
        featureMat = np.concatenate((partFeatureMat,machFeatureMat))
        return featureMat
    
    def stockState(self):      
        priority,deadline,workHour,_ = np.vsplit(self.sche.partInfo,4)
        deadline_cur_hours = (deadline - self.sche.clock)*(deadline > 0) - workHour
        
        slack = -(deadline_cur_hours)/(workHour+e)
        partFeatureMat = np.concatenate((workHour, priority,
                deadline_cur_hours,slack),axis=0) 
        
        width,height = self.feaWidth,self.feaHeight
        return partFeatureMat.reshape(-1,width,height)
    
    def machState(self):
        cur = self.sche.clock
        
        priority = self.sche.machPrio
        hours = (self.sche.workSta - cur) * (self.sche.workSta > 0)
        deadline_cur_hours = (self.sche.deadline - cur)*(self.sche.deadline > 0) - hours
        
        hour_mat = np.zeros(self.stock)
        priority_mat = np.zeros(self.stock)
        deadline_cur_hours_mat = np.zeros(self.stock)
        
        hour_mat[:self.machNum] = hours
        priority_mat[:self.machNum] = priority
        deadline_cur_hours_mat[:self.machNum] = deadline_cur_hours
        
        width,height = self.feaWidth,self.feaHeight
        machFeatureMat = np.vstack((hour_mat,priority_mat,deadline_cur_hours_mat))
        return machFeatureMat.reshape(-1,width,height)

