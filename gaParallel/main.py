# -*- coding: utf-8 -*-
import time

from ga import GeneAlgorithm
from excel import ExcelLog

if __name__ == '__main__':
    testNum = 100
    
    logger = ExcelLog('GA-15-5',True)
    start = time.time()    
    grade_list = []
    ga = GeneAlgorithm()
    for i in range(testNum):
        ga.GaOp.sche.recreate('test')
        ga.iteration(50,400,False)
        grade = ga.GaOp._calIndivi(ga.GaOp.bestIndi,False)
        grade_list.append(grade)
        print(i+1, ' ',grade)
        logger.saveTest(grade)
        # print(ga.GaOp.sche.T)
    average = sum(grade_list)/testNum
    print("num_playouts:, min: {}, average: {}, max:{}".format(
          min(grade_list), 
          average, max(grade_list)))

    end = time.time()
    print("Execution Time: ", end - start)