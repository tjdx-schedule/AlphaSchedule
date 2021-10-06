# -*- coding: utf-8 -*-
import numpy as np

from .EnvirConf import envConfig as ec
from .GenPart import GenPartSeed

class DynamicScheduler:
    def __init__(self, seed):
        self.machNum = ec.MACHNUM
        self.partNum = ec.PARTNUM
        self.stock = ec.STOCKNUM

        self.genPart = GenPartSeed(num = self.partNum,seed = seed)
        self.noopAct = [self.stock, self.machNum]
        
    def reset(self):
        self.clock = 1
        self.countNum = 1
        self.genPart.reset()
        
        self.workSta = np.zeros(self.machNum)
        self.machPrio = np.zeros(self.machNum)
        self.deadline = np.zeros(self.machNum)

        self.partInfo = np.zeros((4,self.stock)) #index 4 is final
        self.partEquCons = np.zeros((self.stock,self.machNum),dtype=np.float32)
        
        self.partMask, self.machMask = self.getMask()
        
        self.lastClock = self.clock
        _,self.newMachFlag = self.getNewEvent()   
     
        self.steps = 0
        self.grade = 0
    
    def step(self,action):
        proceedingFlag = self.machSche(action)
        reward = 0
        if not proceedingFlag:
            reward = self.proceeding()
            self.grade += reward
   
        self.partMask, self.machMask = self.getMask()
        
        done,info = self.isDone()
        self.steps += 1
        return reward,done,info

    def proceeding(self):
        self.clock += 1
        reward = self.getReward()
        newPartState,self.newMachFlag = self.getNewEvent()       
        return reward
    
    def machSche(self,action):#undo:revise 5
        if self.getConstrains(action,'PRE') and \
        self.getConstrains(action,'NOTNOOP') and\
        self.getConstrains(action,'CONS'):
            stockAct, machAct = self.actionTrans(action)
            priority,deadline,workHour,release,= self.partInfo[:,stockAct]
            machIndex = machAct
                        
            self.workSta[machIndex] = workHour + self.clock
            self.machPrio[machIndex] = priority #priority
            self.deadline[machIndex] = deadline

            self.updateStocks(stockAct)
            
            scheFlag = True
            self.newMachFlag -= 1                        
        else:
            scheFlag = False

        return scheFlag  

    def getConstrains(self,action,ruleMode):#undo:revise 4
        stockAct, machAct = self.actionTrans(action)       
        constrains = True        
        if ruleMode == 'PRE':
            constrains = constrains and self.newMachFlag > 0
            constrains = constrains and machAct < self.machNum
            constrains = constrains and stockAct < self.stock
            
        elif ruleMode == 'NOTNOOP':
            constrains = constrains and self.machPrio[machAct] == 0
            constrains = constrains and self.partInfo[0][stockAct] != 0 
            
        elif ruleMode == 'CONS':
            constrains = constrains and self.partEquCons[stockAct,machAct] > 0
        else:
            raise Exception('constrains mode error')
        return constrains
    
        
    def actionTrans(self,action):
        stockAct, machAct = action[0:2]
        return stockAct, machAct
    
    def isDone(self):
        done = False
        info = dict()
        if self.countNum >= self.partNum :
            done = True
            info['episode'] = {'r':round(self.grade/self.partNum,4)}
        return done,info
    
    def getReward(self):#undo:revise 1
        reward = 0
        
        dealIndex = np.where(((self.deadline-self.clock) < 0)&(self.deadline >0))[0]
        deal = self.machPrio[dealIndex].sum()
        waitIndex = np.where(((self.partInfo[1]-self.clock) < 0)&(self.partInfo[1]>0))[0]
        wait = self.partInfo[0,waitIndex].sum()
        reward += -1 * (deal + wait)
        
        return reward
      
    def getNewEvent(self):
        newPart,newPartFlag = self.genPart.getNewPart(self.clock)
        newPartState = 0
        
        if newPartFlag:
            newPartState += 1
            self.lastPartCome = self.clock
            idlePos = np.where(self.partInfo[0] == 0)[0]
            if len(idlePos):
                ranIdlePos = idlePos[0]
                self.updateStocks(ranIdlePos,newPart)
                self.countNum += 1
                
            else:
                newPartFlag = False 
                newPartState += 1
                
        machIndexes = np.where((self.workSta <= self.clock))[0]       
        newMachNum = len(machIndexes)
        if newMachNum > 0:
            self.updateMachs(machIndexes)
        return newPartState,newMachNum#newMachFlag
    
    def updateMachs(self,machIndexes):
        self.workSta[machIndexes] = 0
        self.machPrio[machIndexes] = 0
        self.deadline[machIndexes] = 0
        
    def updateStocks(self,stockIndex, newPart = None):
        if newPart is None:
            self.partInfo[:,stockIndex] = 0
            self.partEquCons[stockIndex,:] = 0
        else:
            self.partInfo[:,stockIndex] = np.array\
            ([newPart.priority,newPart.deadline,newPart.workHour,newPart.release])
            self.partEquCons[stockIndex] = newPart.equCons
  
    def getMask(self):
        machBusyState = (self.machPrio == 0)
        machMask = self.partEquCons * machBusyState 
        
        stockMask = (self.partInfo[0] != 0) & \
        (machMask.sum(1) > 0) 
        return stockMask, machMask


class SkipDynamicScheduler(DynamicScheduler):
    def __init__(self, seed):
        super().__init__(seed=seed)
        
    def reset(self):
        super().reset()
        self.step(self.noopAct)
    
    def step(self, action):
        reward,done,info = super().step(action)
        stepsReward = reward
        while not (done or self.notSkipCon()):
            action = self.noopAct
            reward,done,info = super().step(action)
            
            stepsReward += reward 
        return stepsReward,done,info
            
    def notSkipCon(self):
        stockMask = self.partMask
        # machMask = self.machMask
        # action_mask = machMask * stockMask.reshape(-1,1)
        condition  = stockMask.sum() > 0
        return condition
    
def ranChoose(mask):
    index = np.where(mask!=0)[0]
    if len(index) > 0:
        act = np.random.choice(index)
    else:
        # print('invalid')
        act = len(mask)
    return act

def machRanChoose(mask,partIndex):
    partMax,machMax = mask.shape
    if partIndex == partMax:
        # print('invalid')
        act = machMax
    else:
        act = ranChoose(mask[partIndex])
    return act

if __name__ == '__main__':
    value = 0
    a = SkipDynamicScheduler(0)
    a.reset()
    done, _ = a.isDone()
    while not done:
        partMask, machMask = a.partMask, a.machMask
        act = ranChoose(partMask)
        machAct = machRanChoose(machMask,act)
        reward, done, info = a.step([act,machAct])
        value += reward
    print(info['episode']['r'])
    # print(value/a.partNum)
        
    