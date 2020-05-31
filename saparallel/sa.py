# -*- coding: utf-8 -*-
import math
import numpy as np

from genpart import Scheduler
from config import scheConfig


class SaOperator:
    def __init__(self):
        self.sche = Scheduler()
        self.logger = []

    def NewAnswer(self, S1):
        # S1:the current answer
        # S2:the new answer
        N = len(S1)
        S2 = S1.copy()
        J1, J2 = np.random.choice(range(N), size=2, replace=False)
        W = S2[J1]
        S2[J1] = S2[J2]
        S2[J2] = W
        return S2

    def Metropolis(self, S1, S2, T):
        grade1 = self.sche.scheStatic(S1)
        grade2 = self.sche.scheStatic(S2)
        dG = grade2 - grade1
        if dG < 0 or np.exp(-dG / T) >= np.random.rand():
            S = S2
            grade = grade2
        else:
            S = S1
            grade = grade1
        return S, grade


class SimulatedAnnealing:
    def __init__(self):
        self.T0 = 1000  # start temperature
        self.Tend = 1e-3  # end temperature
        self.L = 38  # iteration times during every  temperature
        self.q = 0.9  # Cooling rate

        self.schConfig = scheConfig
        self.partNum = self.schConfig.partNum

        self.SaOp = SaOperator()

    def run(self):
        S1 = list(range(self.partNum))
        np.random.shuffle(S1)
        count = 0
        gradeLog = []
        answerLog = []
        T=self.T0
        while T > self.Tend:
            gradeTemp = np.zeros(self.L)
            answerTemp = np.zeros((self.L, self.partNum),dtype=np.int)
            for k in range(self.L):
                S2 = self.SaOp.NewAnswer(S1)
                S1,grade=self.SaOp.Metropolis(S1,S2,T)
                gradeTemp[k]=grade
                answerTemp[k]=S1
            tempBestIndex=np.argmin(gradeTemp)
            gradeLog.append(gradeTemp[tempBestIndex])
            answerLog.append(answerTemp[tempBestIndex])
            T=T*self.q
            count = count + 1
            #print(gradeTemp[tempBestIndex],answerTemp[tempBestIndex],count)
        bestIndex=np.argmin(gradeLog)
        bestGrade=gradeLog[bestIndex]
        bestAnswer=answerLog[bestIndex]
        return  bestGrade,bestAnswer


if __name__ == '__main__':
    sa = SimulatedAnnealing()

    sa.SaOp.sche.recreate('test')
    grade,answer=sa.run()
    print("best result is:",grade,answer)
    sa.SaOp.sche.plotGantt()