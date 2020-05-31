# -*- coding: utf-8 -*-
import time

from sa import SimulatedAnnealing

if __name__ == '__main__':
    start = time.time()
    
    grade_list = []
    testNum = 1
    sa = SimulatedAnnealing()
    for i in range(testNum):
        sa.SaOp.sche.recreate('val')
        grade,_ = sa.run()
        grade_list.append(grade)
        print(i, ' ',grade)
    average = sum(grade_list)/testNum
    print("num_playouts:, min: {}, average: {}, max:{}".format(
          min(grade_list), 
          average, max(grade_list)))

    end = time.time()
    print("Execution Time: ", end - start)