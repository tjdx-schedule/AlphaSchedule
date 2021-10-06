# -*- coding: utf-8 -*-
import numpy as np

from .EnvirConf import envConfig 
from .gantt import plotGantt

class GenPart:
    def __init__(self, mode, seed=0):
        self.config = envConfig
        self.partNum = self.config.partNum
        self.machNum = self.config.machNum
    
        self.tight = self.config.tight
        
        self.trainSeed = self.config.trainSeed
        self.testSeed = self.config.testSeed
        self.valSeed = self.config.valSeed
        
        self.maxTime = self.config.maxTime
        self.minTime = self.config.minTime        
        
        self.modeList = ['train','test','val']
        if mode in self.modeList:
            if mode == 'train':
                self.genSeed = self.trainSeed + seed
            elif mode == 'test':
                self.genSeed = self.testSeed - 1 + seed
            elif mode == 'val':
                self.genSeed = self.valSeed - 1 + seed
            self.mode = mode
        else:
            error_str = 'genPart mode error!mode is ' \
                + str(mode)
            raise Exception(error_str)


    def reset(self,update_example):  
        if update_example:
            if self.mode != 'train':
                self.genSeed += 1
        self.setSeed(self.genSeed)
        jobMat,machMat = self.createPart()
               
        return jobMat, machMat

    def setSeed(self, seedValue):
        np.random.seed(seedValue)
        
    def createPart(self):
        jobTimeMat = np.random.randint(self.minTime, self.maxTime, self.partNum)
        tightMat = jobTimeMat * (1 + np.random.rand(self.partNum) * self.tight)
        priorityMat = np.random.randint(1,self.config.priority, self.partNum)
        
        jobTimeMat = np.vstack((jobTimeMat,np.floor(tightMat),priorityMat)).T
        
        machMat = np.zeros(self.machNum)
        return jobTimeMat, machMat
    
class Scheduler:
    def __init__(self, mode, seed):
        self.genpart = GenPart(mode, seed)
        
    def reset(self,update_example):
        self.T, self.M = self.genpart.reset(update_example)
        
        self.part = np.copy(self.T)
        self.mach = np.copy(self.M)
        
        self.partScheLog = np.zeros((self.genpart.partNum,5))
        self.sche_count = self.genpart.partNum 
        #start,end,machine
        # self.machLog = [[] for i in range(self.genpart.machNum)]
        
    def step(self, partIndex):
        # partIndex = int(partIndex)
        #get
        hours, deadline, priority = self.part[partIndex]
        machIndex = np.argmin(self.mach)
        
        #cal
        start = self.mach[machIndex]
        end = start + hours
        grade = deadline - end
        
        #set
        self.mach[machIndex] = end
        self.part[partIndex] = 0
        
        #log
        self.partScheLog[partIndex] = [start, end, machIndex, grade, priority]
        self.sche_count -= 1

    def is_end(self):
        flag, grade = False, None
        if self.sche_count == 1:
            partIndex = self.available()
            self.step(partIndex[0])
        hourSum = np.sum(self.part[:,0])
        if hourSum == 0:
            flag = True
            grade = self.getGrade()
        return flag, grade
    
    def available(self):
        avai = np.where(self.part[:,0] != 0)[0]
        return avai
    
    def getGrade(self):
        gradeMat = self.partScheLog[:,3]
        priorityMat = self.partScheLog[:,4]
        
        filterIndex = np.where(gradeMat < 0)[0]
        grade = -np.sum(gradeMat[filterIndex]*priorityMat[filterIndex])
        return grade
        
    def plotGantt(self):
        end, _ = self.is_end()
        if end:
            plotGantt(self.partScheLog,self.genpart.machNum)
        else:
            raise Exception('plot but not end')    

    def scheStatic(self, actLi):
        i = 0
        self.reset(False)
        done, _ = self.is_end()
        while not done:
            act = actLi[i]
            self.step(act)
            done, grade = self.is_end()
            
            i += 1
        return grade
            
if __name__ == '__main__':
    a = Scheduler('val',0)
    sumFlag = 0
    for i in range(1):
        print(i)
        actLi = []
        a.reset(True)
        done, _ = a.is_end()
        while not done:
            act = np.random.choice(a.available())
            a.step(act)
            actLi.append(act)
            done, grade = a.is_end()
        print('grade = ', grade)
        a.plotGantt() 