# -*- coding: utf-8 -*-
import math
import numpy as np

from genpart import Scheduler
from config import scheConfig

c1 = 2
c2 =2.1
Wstart = 0.9
Wend = 0.4
popuSize = 100 
iterationTimes = 100
class PsoOperator:
    def __init__(self, popsize, chromlen, scheduler):
        self.sche = scheduler
        self.loggerReset(popsize, chromlen)

    def loggerReset(self, popsize, chromlen):
        self.indiBestAnswer = np.zeros((popsize, chromlen),dtype=np.int)
        self.indiBestGrade=1/(np.zeros(popsize)+(1e-9))

        self.globalBestGrade = np.float('inf')
        self.globalBestIndi = np.zeros(chromlen,dtype=np.int)

    def _calIndivi(self, individual, isPlot=False):
        indi = individual.tolist()
        # print(indi)
        grade = self.sche.scheStatic(indi)
        if isPlot:
            self.sche.plotGantt()
        return grade


class ParticleSwarmOptimization:
    def __init__(self, c1, c2, Wstart, Wend, popuSize, scheduler):
        # Acceleration factor
        self.c1 = c1
        self.c2 = c2
        self.Wstart = Wstart
        self.Wend = Wend 
        
        self.alpha = c1 + c2
        self.k = 1 - 1/self.alpha + math.sqrt(abs(self.alpha**2 - 4*self.alpha))/2

        self.popuSize = popuSize
        self.schConfig = scheConfig
        self.partNum = self.schConfig.partNum
        self.chromLen = self.partNum

        self.PsoOp = PsoOperator(self.popuSize, self.chromLen, scheduler)

        self.population = np.zeros((self.popuSize, self.chromLen),dtype=np.int)
        self.speed = np.zeros((self.popuSize, self.chromLen))

    def initPopulation(self):
        self.population = np.zeros((self.popuSize, self.chromLen),dtype=np.int)
        self.speed = np.zeros((self.popuSize, self.chromLen))
        for m in range(self.popuSize):
            seq = list(range(self.partNum))
            np.random.shuffle(seq)
            self.population[m] = seq

    def speedRefresh(self):
        r1, r2 = np.hsplit(np.random.rand(self.popuSize, 2), 2)

        self.speed = self.speed * self.W \
            + self.c1 * r1 * (self.PsoOp.indiBestAnswer - self.population) \
            + self.c2 * r2 * (self.PsoOp.globalBestIndi - self.population)

    def popRefresh(self):
        populationMiddle=self.population +  self.speed
        for m in range(self.popuSize):
            self.population[m]=np.argsort(np.argsort(populationMiddle[m]))

    def indiBestRefresh(self):
        for m in range(self.popuSize):
            grade = self.PsoOp._calIndivi(self.population[m])
            if grade<self.PsoOp.indiBestGrade[m]:
                self.PsoOp.indiBestAnswer[m]=self.population[m]
                self.PsoOp.indiBestGrade[m]=grade

    def globalBestRefresh(self):
        bestIndex=np.argmin(self.PsoOp.indiBestGrade)
        self.PsoOp.globalBestGrade=self.PsoOp.indiBestGrade[bestIndex]
        self.PsoOp.globalBestIndi=self.PsoOp.indiBestAnswer[bestIndex]
    
    def weightUpdate(self, iterNum, iterMax):
        ws = self.Wstart
        we = self.Wend
        self.W = we+(ws-we)*(iterMax-iterNum)/iterMax
        
    def iteration(self,iterNum):
        self.PsoOp.loggerReset(self.popuSize,self.chromLen)
        for i in range(iterNum):
            self.indiBestRefresh()
            self.globalBestRefresh()
            
            self.weightUpdate(i,iterNum)
            
            self.speedRefresh()
            self.popRefresh()

if __name__ == '__main__':
    pso=ParticleSwarmOptimization(c1, c2, Wstart, Wend, popuSize)
    pso.initPopulation()
    pso.PsoOp.sche.recreate('test')
    pso.iteration(100)
    print(pso.PsoOp.globalBestGrade,pso.PsoOp.globalBestIndi)
