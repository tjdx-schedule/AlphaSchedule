# -*- coding: utf-8 -*-
import time

from sa import SimulatedAnnealing
from excel import ExcelLog

if __name__ == '__main__':
    testNum = 100
    
    logger = ExcelLog('SATest35_15_s',True)
    start = time.time()    
    grade_list = []
    sa = SimulatedAnnealing()
    for i in range(testNum):
        sa.SaOp.sche.recreate('test')
        grade, _= sa.run()
        grade_list.append(grade)
        print(i+1, ' ',grade)
        logger.saveTest(grade)
        #print(sa.SaOp.sche.T)
    average = sum(grade_list)/testNum
    print("num_playouts:, min: {}, average: {}, max:{}".format(
          min(grade_list), 
          average, max(grade_list)))

    end = time.time()
    print("Execution Time: ", end - start)