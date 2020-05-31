# -*- coding: utf-8 -*-
import math
import numpy as np

from genpart import Scheduler 
from config import scheConfig

class GaOpeartor:
    def __init__(self):    
        self.sche = Scheduler()
        self.loggerReset()
    
    def loggerReset(self):
        self.logger = []
        
        self.bestGrade = np.float('inf')
        self.bestIndi = None
    
    def roulette(self, popu, crossSize, n, N, isPrint=False):
        popuSize = popu.shape[0]
        
        grade, fitness = self._calFitness(popu, n, N, True)
        self.keepElite(popu,grade,n,isPrint)
       
        prob = fitness/np.sum(fitness)  
        # print(prob)
        chooseIndex = np.random.choice(popuSize,
                                       size=crossSize,
                                       p=prob,
                                       replace=True)
        return chooseIndex
    
    def _calFitness(self, popu, n=0, N=0, need_fit=False):
        popuSize = popu.shape[0]
        
        gradeArr = np.zeros(popuSize)
        for i in range(popuSize):
            gradeArr[i] = self._calIndivi(popu[i])      
        fitness = None
        if need_fit:
            m = 1 + math.log(N)
            n = min(n+1,N)
            fitness = int(n**(1/m))/gradeArr
        return gradeArr,fitness
    
    def _calIndivi(self, individual,isPlot = False):
        indi = individual.tolist()
        grade = self.sche.scheStatic(indi)
        if isPlot:
            self.sche.plotGantt()
        return grade
    
    def keepElite(self,popu,grade,n,isPrint):       
        iterBest = np.min(grade)
        if iterBest <= self.bestGrade:
            self.bestIndi = popu[np.argmin(grade)]
            self.bestGrade = iterBest
            
        argSortIndex = np.argsort(grade)[0:5]
        nDict = {'iter':n,'indivis':popu[argSortIndex],
                 'grades':grade[argSortIndex],'currentBest':self.bestGrade}
        self.logger.append(nDict)
        if isPrint:
            print('iter:',n,' bestGrade:',self.bestGrade,' iterBest:',grade[argSortIndex])
            
    def crossOperator(self,p1,p2,J1,J2):
        c1 = self.crossover(p1,p2,J1)
        c2 = self.crossover(p2,p1,J2)
        return c1,c2
    
    def crossover(self,p1,p2,J):
        p1FillIndex = np.where(p1==J)[0]
        p1DragIndex = np.where(p1!=J)[0]
        
        p2DragIndex = np.where(p2!=J)[0]
        p2DragContent = p2[p2DragIndex]
        
        c = np.zeros_like(p1)
        c[p1FillIndex] = J
        c[p1DragIndex] = p2DragContent
        return c
    
    def mutateOperator(self,p,J1,J2):
        J1Index = np.where(p == J1)[0]
        J2Index = np.where(p == J2)[0]
        if J1Index[0] < J2Index[0]:
            first = J2
            second = J1
            firstLen = J2Index.shape[0]
        else:
            first = J1
            second = J2
            firstLen = J1Index.shape[0]
        index = np.concatenate((J1Index,J2Index))
        index.sort()
        p[index[:firstLen]] = first
        p[index[firstLen:]] = second
        return p           

class GeneAlgorithm:
    def __init__(self):
        self.mutateRate = 0.1
        self.crossRate = 0.6
        self.gap = 0.6
                        
        self.schConfig = scheConfig
        self.partNum = self.schConfig.partNum
        # self.orderNum = self.schConfig.orderNum
        self.chromLen = self.partNum
        
        self.GaOp = GaOpeartor()
        
    def initPopulation(self,popuSize):
        self.popuSize = popuSize if popuSize % 2 == 0 else (popuSize+1)
        self.population = np.zeros((self.popuSize,self.chromLen),dtype = np.int)
        for m in range(self.popuSize):
            seq = list(range(self.partNum))
            np.random.shuffle(seq)
            self.population[m] = seq
    
    def iteration(self,popuSize,iterNum,isPrint=False):
        self.GaOp.loggerReset()
        self.initPopulation(popuSize)
        for i in range(iterNum):
            self.evalPopu(i,iterNum,isPrint)
            self.cross()
            self.mutate()
            self.elitePolicy()
        self.GaOp.roulette(self.population,self.popuSize,iterNum,iterNum,isPrint)
        
    def evalPopu(self, i, maxIter, isPrint):
        popuIndex = self.GaOp.roulette(np.copy(self.population),self.popuSize,\
                                        i,maxIter,isPrint)
        self.tempPopu = self.population[popuIndex]
            
    def cross(self):
        crossProb = np.random.rand(self.popuSize//2)
        crossIndex = np.where(crossProb <= self.crossRate)[0]
        for i in crossIndex:
            index = [2*i,2*i+1]
            p1,p2 = self.tempPopu[index]
            J1,J2 = np.random.choice(range(self.partNum),size=2,replace=False)#
            c1,c2 = self.GaOp.crossOperator(p1,p2,J1,J2)
            self.tempPopu[index] = [c1,c2]
    
    def mutate(self):
        ranProb = np.random.rand(self.popuSize)
        mutateIndex = np.where(ranProb <= self.mutateRate)[0]
        for index in mutateIndex:
            J1, J2 = np.random.choice(range(self.partNum),size=2,replace=False)
            c = self.GaOp.mutateOperator(self.tempPopu[index],J1,J2)
            self.tempPopu[index] = c
                   
    def elitePolicy(self):
        # self.population[0] = self.GaOp.bestIndi
        # self.population[0:self.GaOp.eliteNum] = self.GaOp.eliteGroup
        self.population = self.tempPopu.copy()
    
if __name__ == '__main__':
    ga = GeneAlgorithm()
    
    ga.GaOp.sche.recreate('test')
    ga.iteration(50,400,True)
    ga.GaOp._calIndivi(ga.GaOp.bestIndi,True)