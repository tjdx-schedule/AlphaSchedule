# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 12:44:39 2019

@author: Administrator
"""
import numpy as np
from .EnvirConf import envConfig as ec

class GenPartSeed:
    def __init__(self,num = 600,seed = 0):

        self.num = num * 2
        
        self.setSeed(seed)
    
    def reset(self):
        if self.mode == 'TEST' or self.mode == 'VALID':
            np.random.seed(self.seed)                 
            self.seed += 1                    
        self.tic = 1
        self.reCreate()
    
    def reCreate(self):
        self.interval = np.random.randint(ec.COMEMIN,ec.COMEMAX,size = self.num).tolist()
        self.workHourLi = np.random.randint(ec.WORKHOURMIN,ec.WORKHOURMAX,size = self.num).tolist()
        self.priorityLi = np.random.randint(1,ec.PRIORITYMAX,size = self.num).tolist()
        self.tightLi = np.random.random(size = self.num).tolist()
        
        machNum = ec.MACHNUM
        useFlag = 1
        self.equStockConsNum = np.random.randint(1,machNum+1,size = self.num)
        self.equStockCons = np.zeros((self.num,machNum))
        for i in range(self.num):
            equConNum = self.equStockConsNum[i]
            ranSeq = np.random.choice(a=machNum, size=equConNum, replace=False, p=None)
            self.equStockCons[i,ranSeq] = useFlag
        self.equStockCons = self.equStockCons.tolist()
    
                        
    def pause(self):
        self.tic = float('inf')
    
    def getNewPart(self,clock): 
        #t1 = np.random.exponential(5)
        part = []
        newPartFlag = False
        if clock >= self.tic:
            part = self.createNew(clock)
            self.tic += self.interval.pop(0)#(int(np.random.exponential(self.y)) + 1)
            newPartFlag = True
        return part,newPartFlag            
    
    def createNew(self,release):
        if len(self.tightLi) < 1:
            self.reCreate()
        workHour = self.workHourLi.pop(0)#(int(np.random.exponential(u)) + 1)#np.random.randint(10,100)
        tight = 1 + ec.TIGHT * self.tightLi.pop(0)
        deadline = int(workHour * tight) + release
        priority = self.priorityLi.pop(0)
        equCons = self.equStockCons.pop(0)
        newPart = ParallelOpera(workHour,release,priority,deadline,equCons)
        return newPart
   
    def setSeed(self,seed):
        if seed < ec.EVALSEED:
            self.mode = 'TRAIN'
        elif seed < ec.TESTSEED:
            self.mode = 'VALID'
        else:
            self.mode = 'TEST'             
        self.seed = seed
        np.random.seed(seed)

class GenConsSeed:
    def __init__(self,name,machNum,consNum,seed):
        self.name = name
        self.consNum = consNum        
        self.machNum = machNum
        self.seed = seed
        
        self._consMat = self.initConsMat()
        
    def initConsMat(self):
        consNum,machNum = self.consNum,self.machNum 
        consMat = np.zeros((consNum,machNum))
        
        np.random.seed(self.seed)
        extraConsMachNum = np.random.randint(1,machNum+1,size = consNum)
        
        minNum = min(consNum,machNum)
        consMat[:minNum,:minNum] = np.identity(minNum)
        if consNum < machNum:
            ranSeq = np.random.choice(consNum,machNum-consNum)
            consMat[ranSeq,range(consNum,machNum)] = 1

        
        for i in range(consNum):
            shouldUseNum = extraConsMachNum[i] 
            alreadyNum = consMat[i].sum()
            
            delta = shouldUseNum - alreadyNum.sum()
            if delta > 0:
                tempIndex = np.where(consMat[i] == 0)[0]
                sampleIndex = np.random.choice(len(tempIndex),int(delta),replace = False)
                sampleTemp = tempIndex[sampleIndex]
                
                consMat[i,sampleTemp] = 1
        
#        self.consMat = consMat
        return consMat
    
    @property
    def consMat(self):
        return self._consMat
    
        
class ParallelOpera:
    def __init__(self,workHour,release,priority,deadline,equCons):
        self.workHour = workHour
        self.release = release
        self.priority = priority
        self.deadline = deadline

        self.equCons = equCons
        self.consUseNum = sum(equCons)
#        self.overtime = overtime
        
        
        
        