# -*- coding: utf-8 -*-
import math
import numpy as np

from genpart import Scheduler 
from config import scheConfig

class GaOpeartor:
    def __init__(self):       
        self.sche = Scheduler()
        self.logger = []
        
        self.bestGrade = np.float('inf')
        self.bestIndi = None
    
    def roulette(self, popu, crossSize, n, N):
        popuSize = popu.shape[0]
        
        grade, fitness = self._calFitness(popu, n, N, True)
        self.keepElite(popu,grade,n)
       
        prob = fitness/np.sum(fitness)  
        # print(prob)
        chooseIndex = np.random.choice(popuSize,
                                       size=(int(crossSize/2),2),
                                       p=prob)
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
    
    def keepElite(self,popu,grade,n):       
        iterBest = np.min(grade)
        if iterBest <= self.bestGrade:
            self.bestIndi = popu[np.argmin(grade)]
            self.bestGrade = iterBest
        
        argSortIndex = np.argsort(grade)[0:5]
        nDict = {'iter':n,'indivis':popu[argSortIndex],
                 'grades':grade[argSortIndex],'currentBest':self.bestGrade}
        self.logger.append(nDict)
        
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
        self.mutateRate = 0.05
        self.crossRate = 0.6
        self.gap = 0.6
        
        self.popuSize = 200
        crossSize = int(self.popuSize * self.gap)
        self.crossSize = crossSize if crossSize % 2 == 0 else crossSize + 1
                
        self.schConfig = scheConfig
        self.partNum = self.schConfig.partNum
        self.orderNum = self.schConfig.orderNum
        self.chromLen = self.partNum * self.orderNum
        
        self.GaOp = GaOpeartor()
        
    def initPopulation(self):
        self.population = np.zeros((self.popuSize,self.chromLen),dtype = np.int)
        for m in range(self.popuSize):
            seq = []
            for i in range(self.partNum):
                seq.extend([i]*self.orderNum)
            np.random.shuffle(seq)
            self.population[m] = seq
    
    def iteration(self,iterNum):      
        self.initPopulation()
        for i in range(iterNum):
            self.cross(i,iterNum)
            self.mutate()
            self.elitePolicy()
        self.GaOp.roulette(self.population,self.crossSize,iterNum,iterNum)
        
    
    def cross(self, i, maxIter):
        crossIndex = self.GaOp.roulette(np.copy(self.population),self.crossSize,i,maxIter)
        for i in crossIndex:
            p1,p2 = self.population[i]
            J1,J2 = np.random.choice(range(self.partNum),size=2,replace=False)#
            c1,c2 = self.GaOp.crossOperator(p1,p2,J1,J2)
            self.population[i] = [c1,c2]
    
    def mutate(self):
        ranProb = np.random.rand(self.popuSize)
        mutateIndex = np.where(ranProb <= self.mutateRate)[0]
        for index in mutateIndex:
            J1, J2 = np.random.choice(range(self.partNum),size=2,replace=False)
            c = self.GaOp.mutateOperator(self.population[index],J1,J2)
            self.population[index] = c
                   
    def elitePolicy(self):
        # ranIndex = np.random.randint(self.popuSize)
        self.population[0] = self.GaOp.bestIndi

    
if __name__ == '__main__':
    ga = GeneAlgorithm()
    ga.GaOp.sche.recreate('val')
    ga.iteration(50)
    ga.GaOp._calIndivi(ga.GaOp.bestIndi,True)