# -*- coding: utf-8 -*-
import time

from excel import ExcelLog
from ga import GeneAlgorithm

if __name__ == '__main__':
    start = time.time()
    logger = ExcelLog('GA-88',True)
    
    grade_list = []
    testNum = 50
    ga = GeneAlgorithm()
    for i in range(testNum):
        ga.GaOp.sche.recreate('val')
        ga.iteration(50)
        grade = ga.GaOp._calIndivi(ga.GaOp.bestIndi,False)
        grade_list.append(grade)
        print(i, ' ',grade)
        logger.saveTest(grade)
    average = sum(grade_list)/testNum
    print("num_playouts:, min: {}, average: {}, max:{}".format(
          min(grade_list), 
          average, max(grade_list)))

    end = time.time()
    print("Execution Time: ", end - start)