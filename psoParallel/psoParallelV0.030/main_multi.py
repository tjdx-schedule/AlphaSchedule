# -*- coding: utf-8 -*-
from config import scheConfig,load
import time

from genpart import Scheduler
import numpy as np
from multiAgent import Agent
from excel import ExcelLog


c1 = 2
c2 =2.1
Wstart = 0.9
Wend = 0.4
popuSize = 100 
subPopNum = 2
subPopSize = np.int(popuSize/subPopNum)
iterationTimes = 100

if __name__ == '__main__':
    testNum = 100
    scheduler = Scheduler()
    BA = Agent(c1, c2, Wstart, Wend, popuSize, scheduler)
    SA = Agent(c1, c2, Wstart, Wend, popuSize, scheduler)
    EA1 = Agent(c1, c2, Wstart, Wend, subPopSize, scheduler)
    EA2 = Agent(c1, c2, Wstart, Wend, subPopSize, scheduler)
    logger = ExcelLog('PSOTest' + str(scheConfig.partNum) + '-' + str(scheConfig.machNum) + load,True)
    start = time.time()    
    grade_list = []
    for i in range(testNum):
        scheduler.recreate('test')
        BA.swarm.initPopulation()
        BA.sendSwarm(SA)
        SA.dividePop(EA1, EA2)
        EA1.runPSO(iterationTimes)
        EA2.runPSO(iterationTimes)
        SA.migrate(EA1,EA2)
        BA.receiveBest(EA1, EA2)
        # print(EA1.swarm.population, EA2.swarm.population)
        BA.runPSO(iterationTimes)
        grade = BA.swarm.PsoOp.globalBestGrade
        grade_list.append(grade)
        print(i+1, ' ',grade)
        logger.saveTest(grade)
        # print(pso.PsoOp.sche.T)
    average = sum(grade_list)/testNum
    print("num_playouts:, min: {}, average: {}, max:{}".format(
          min(grade_list), 
          average, max(grade_list)))

    end = time.time()
    print("Execution Time: ", end - start)