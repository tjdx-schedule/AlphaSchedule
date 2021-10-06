# -*- coding: utf-8 -*-
import numpy as np

from a2c_ppo_acktr.game.scheduler import SkipDynamicScheduler
from a2c_ppo_acktr.game.EnvirConf import envConfig as ec
from a2c_ppo_acktr.game.ExcelLog import ExcelLog

e = 1e-10

def ruleChoose(env,ruleName):
    partMask, machMaskMat = env.partMask, env.machMask
    cur = env.clock
    
    partIndexes = np.where(partMask > 0)[0]
    
    priority,deadline,hours,release = np.vsplit(env.partInfo[:,partIndexes],4)
    deadline_cur = (deadline - cur)*(deadline > 0)
    deadline_cur_hours = deadline_cur - hours
    priority_hours = priority/(hours+e)
    
    if ruleName == 'wspt':
        coef = -hours/(priority+e)
    if ruleName == 'wmdd':
        coef = -np.maximum(hours,deadline_cur)/(priority+e)
    if ruleName == 'atc':
        coef = np.exp(-np.maximum(deadline_cur_hours,0)/(ec.h*hours+e))*priority_hours
    if ruleName == 'wcovert':
        coef = np.maximum(1-np.maximum(deadline_cur_hours,0)/(ec.Kt*hours+e),0)*priority_hours 
    index = np.argmax(coef)
    partAct = partIndexes[index]
    
    machMask = machMaskMat[partAct]
    machIndexes = np.where(machMask >0)[0]
    partEquCons = env.partEquCons[:,machIndexes].sum(0)
    # print(partEquCons)
    machIndex = np.argmin(partEquCons)
    # print(machIndexes)
    machAct = machIndexes[machIndex]
    
    act = [partAct,machAct]
    return act

if __name__ == '__main__':        
    testNum = 100
    # ruleDict = ['wspt','wmdd','atc','wcovert']
    ruleDict =['wmdd']
    
    env = SkipDynamicScheduler(ec.TESTSEED)
    for name in ruleDict:
        logger = ExcelLog(name=name,isLog=False)
        gradeLi = []
        for i in range(testNum):
            env.reset()
            done = False
            while not done:
                act = ruleChoose(env,name)
                reward, done, info = env.step(act)
            grade = info['episode']['r']
            gradeLi.append(grade)
            print(name,' ',i+1,' ',grade)
            logger.saveTest(grade)
        print('------average:',round(sum(gradeLi)/testNum,4))