import numpy as np

from excel import ExcelDeal
class ScheConfig:
    def __init__(self):
        ex = ExcelDeal()
        
        self.partMat,self.machMat = ex.getPaLi('ft10')
        self.partNum,self.orderNum = self.partMat.shape

class RandomConfig:
    def __init__(self):
        self.partNum, self.machNum =35,15
        self.tight = 0.5
        self.priority = 10
        
        self.maxTime = 100
        self.minTime = 5
        
        self.testSeed = 0
        self.valSeed = 1000
        self.trainSeed = np.random.randint(2000,10000)
        
        self.period = 50

scheConfig = RandomConfig()