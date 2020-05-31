# -*- coding: utf-8 -*-
import numpy as np

from config import scheConfig
from gantt import plotGantt

class GenPart:
    def __init__(self):
        self.config = scheConfig
        self.partNum = self.config.partNum
        self.orderNum = self.config.orderNum
        
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
        jobTimeMat = np.random.randint(self.minTime, self.maxTime ,(self.partNum,self.orderNum))
        machMat = []
        for i in range(self.partNum):
            equs = np.arange(self.orderNum)
            np.random.shuffle(equs)
            machMat.append(equs)
        return jobTimeMat, np.array(machMat)
            
class Scheduler:
    def __init__(self):
        self.genpart = GenPart()
        
        self.maxConst = 10e4
        self.spInitSize = self.genpart.orderNum * 10
        self.spDelta = self.genpart.orderNum
     
    def recreate(self, mode):
        self.genpart.recreate(mode)
    
    def reset(self):
        self.T, self.M = self.genpart.reset()
        
        self.partOrder = np.zeros(self.genpart.partNum,dtype = np.int)
        self.partLastTime = np.zeros(self.genpart.partNum)
        self.partScheLog = np.zeros((self.genpart.partNum,self.genpart.orderNum,2))
        
        self.machScheLog = np.zeros((self.genpart.orderNum,self.genpart.partNum,4))
        self.machScheLog[:,:,-2:] = -1
        self.machOrder = np.zeros(self.genpart.orderNum, dtype = np.int)
        
        self.idleCount = self.spInitSize * np.ones(self.genpart.orderNum,dtype = np.int)# sp Num
        
        self.machIdleMat = [] #sp: idle Time Log 
        for i in range(self.genpart.orderNum):
            idleArr = np.zeros((2,self.idleCount[i]))
            idleArr[1] = self.maxConst
            self.machIdleMat.append(idleArr)
        # return self.is_end()
        
    def step(self, partIndex):
        # get 
        tOrder = self.partOrder[partIndex]
        hours = self.T[partIndex,tOrder]
        tMach = self.M[partIndex,tOrder] 
        tStart = self.partLastTime[partIndex]
        
        tMachIdle = self.machIdleMat[tMach]
        tMachSpSize = self.idleCount[tMach]
        
        machOrderNum = self.machOrder[tMach]
        
        #find
        for i in range(tMachSpSize):
            idleStart, idleEnd = tMachIdle[:,i]
            idleHour = idleEnd - idleStart
           
            if idleEnd > tStart and min(idleEnd-tStart,idleHour) >= hours:
                # insert
                tStart_norm = max(tStart, idleStart)
                tEnd_norm = tStart_norm + hours
                if tStart_norm > idleStart and  tEnd_norm < idleEnd: #noLimit idle
                    # 1 to 2
                    firstStart, firstEnd = idleStart, tStart_norm
                    secondStart, secondEnd = tEnd_norm, idleEnd
                    self.machIdleMat[tMach][:,i+2:] = self.machIdleMat[tMach][:,i+1:-1]#create space
                    self.machIdleMat[tMach][:,i] = [firstStart, firstEnd]
                    self.machIdleMat[tMach][:,i+1] = [secondStart, secondEnd]#insert
                elif idleStart  == tStart_norm and tEnd_norm == idleEnd : #1 to 0
                    self.machIdleMat[tMach][:,i:-1] = self.machIdleMat[tMach][:,i+1:]
                elif idleStart  == tStart_norm: # 1 to 1   noLimit idle
                    self.machIdleMat[tMach][:,i] = [tEnd_norm, idleEnd]
                elif tEnd_norm == idleEnd:#1 to 1
                    self.machIdleMat[tMach][:,i] = [idleStart, tStart_norm]
                    # self.machIdleMat[tMach][0,i+1] = tEnd_norm
                else:
                    raise Exception('error')
                
                break
        
        #set
        self.partLastTime[partIndex] = tEnd_norm
        self.partScheLog[partIndex,tOrder] = [tStart_norm,tEnd_norm]
        self.machScheLog[tMach,machOrderNum] = [tStart_norm,tEnd_norm,partIndex,tOrder]
        
        self.partOrder[partIndex] += 1
        self.machOrder[tMach] += 1
        
        # return self.is_end()
               
    def is_end(self):
        end_flag = self.partOrder.sum() == (self.genpart.partNum * self.genpart.orderNum)
        grade = None
        if end_flag:
            grade = self.partLastTime.max()
            
        return end_flag, grade
    
    def available(self):
        avai = np.where(self.partOrder != self.genpart.orderNum)[0]
        return avai
 
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
          
    def plotGantt(self):
        end, _ = self.is_end()
        if end:
            plotGantt(self.partScheLog,self.M)
        else:
            raise Exception('plot but not end')

    def _valMach(self):
        partNum, machNum = self.genpart.partNum, self.genpart.orderNum
        machLog = [[] for i in range(self.genpart.orderNum)]
        
        for i in range(partNum):
            for j in range(machNum):
                machIndex = self.M[i,j]
                start,end = self.partScheLog[i,j]
                machLog[machIndex].append([start,end])
        
        flag = True
        for machLi in machLog:
            machLi.sort(key=lambda x:x[0])
            for k in range(len(machLi)-1):
                flag &= machLi[k+1][0]>= machLi[k][1]
                if not flag:
                    print(k,machLi[k+1][0],machLi[k+1][1])
                    break
        print('flag =', flag)
        return flag

    def _valMachLog(self):
        partNum, machNum = self.genpart.partNum, self.genpart.orderNum
        machLogStart = [[] for i in range(self.genpart.orderNum)]
        machLogEnd = [[] for i in range(self.genpart.orderNum)]
        for i in range(partNum):
            for j in range(machNum):
                machIndex = self.M[i,j]
                start,end = self.partScheLog[i,j]
                machLogStart[machIndex].append(start)
                machLogEnd[machIndex].append(end)
        
        for i in range(machNum):
            machStartLi = machLogStart[i]
            machEndLi = machLogEnd[i]
            machStartLi.sort()
            machEndLi.sort()
        machLogStart = np.array(machLogStart)
        machLogEnd = np.array(machLogEnd)
        
        a = (machLogStart == self.machScheLog[:,:,0]).sum()
        b = (machLogEnd == self.machScheLog[:,:,1]).sum()
        print(machLogStart)
        print(self.machScheLog[:,:,2])
        print(a,b)

if __name__ == '__main__':
    a = Scheduler()
    sumFlag = 0
    for i in range(1):
        print(i)
        actLi = []
        a.reset()
        done, _ = a.is_end()
        while not done:
            act = np.random.choice(a.available())
            actLi.append(str(act)+'-'+str(a.partOrder[act]))
            a.step(act)
            done, _ = a.is_end()
        print('grade = ', a.partLastTime.max())
        plotGantt(a.partScheLog,a.M)   
        
        a._valMach()
        # sumFlag += int(a._valMach())    
        
        
        