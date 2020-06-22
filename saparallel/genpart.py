# -*- coding: utf-8 -*-
import numpy as np

from config import scheConfig
from gantt import plotGantt

class GenPart:
    def __init__(self):
        self.config = scheConfig
        self.partNum = self.config.partNum
        self.machNum = self.config.machNum
    
        self.tight = self.config.tight
        
        self.trainSeed = self.config.trainSeed
        self.testSeed = self.config.testSeed
        self.valSeed = self.config.valSeed
        
        self.maxTime = self.config.maxTime
        self.minTime = self.config.minTime      
        
        self.jobMat = None
        self.machMat = None
        
        self.mode = 'train'
        
    def reset(self):   
        # jobMat,machMat = np.copy(self.config.partMat),np.copy(self.config.machMat)
        if self.jobMat is None :
            self.recreate()
        jobMat, machMat = self.jobMat, self.machMat     
        return jobMat, machMat
    
    def recreate(self, mode):
        if mode == 'train':
            self.trainSeed += 1                
            seed = self.trainSeed
        elif mode == 'val':
            if mode == self.mode:
                self.valSeed += 1
            else:
                self.valSeed = self.config.valSeed
            seed = self.valSeed
        elif mode == 'test':
            if mode == self.mode:
                self.testSeed += 1
            else:
                self.testSeed = self.config.testSeed
            seed = self.testSeed
        else:
            raise Exception('mode error')
        self.mode = mode
        self.setSeed(seed)
        
        self.jobMat, self.machMat = self.createPart()
        # print(self.jobMat)
    
    def setSeed(self, seedValue):
        np.random.seed(seedValue)
        self.seed = seedValue
        
    def createPart(self):
        jobTimeMat = np.random.randint(self.minTime, self.maxTime, self.partNum)
        tightMat = jobTimeMat * (1 + np.random.rand(self.partNum) * self.tight)
        priorityMat = np.random.randint(1,self.config.priority, self.partNum)
        
        jobTimeMat = np.vstack((jobTimeMat,np.floor(tightMat),priorityMat)).T
        
        machMat = np.zeros(self.machNum)
        return jobTimeMat, machMat
    
class Scheduler:
    def __init__(self):
        self.genpart = GenPart()
    
    def recreate(self, mode):
        self.genpart.recreate(mode)
    
    def reset(self):
        self.T, self.M = self.genpart.reset()
        
        self.part = np.copy(self.T)
        self.mach = np.copy(self.M)
        
        self.partScheLog = np.zeros((self.genpart.partNum,5))
        #start,end,machine
        # self.machLog = [[] for i in range(self.genpart.machNum)]
        
    def step(self, partIndex):
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

    def is_end(self):
        flag, grade = False, None
        
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
        self.reset()
        done, _ = self.is_end()
        while not done:
            act = actLi[i]
            self.step(act)
            done, grade = self.is_end()
            
            i += 1
        return grade

if __name__ == '__main__':
    a = GenPart()
    a.recreate('val')
    b,c = a.reset()
        
        # sumFlag += int(a._valMach())    
        
        
        