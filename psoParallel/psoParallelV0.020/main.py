# -*- coding: utf-8 -*-
import time

from pso import ParticleSwarmOptimization
from excel import ExcelLog

if __name__ == '__main__':
    testNum = 100
    
    logger = ExcelLog('PSOTest20_l_7065',True)
    start = time.time()    
    grade_list = []
    pso=ParticleSwarmOptimization()
    for i in range(testNum):
        pso.PsoOp.sche.recreate('test')
        pso.iteration(60)
        grade = pso.PsoOp._calIndivi(pso.PsoOp.globalBestIndi,False)
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