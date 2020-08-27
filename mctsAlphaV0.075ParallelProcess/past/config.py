import numpy as np

from excel import ExcelDeal
class ScheConfig:
    def __init__(self):
        ex = ExcelDeal()
        
        self.partMat,self.machMat = ex.getPaLi('ft10')
        self.partNum,self.orderNum = self.partMat.shape

class ObsConfig:
    def __init__(self):
        self.stackFrameNum = 1
        self.ruleFeatureNum = 5
        
        self.width = 11
        self.height = 5

class RandomConfig:
    def __init__(self):
        self.partNum, self.machNum =55,25
        self.tight = 0.65
        self.priority = 14
        
        self.maxTime = 150
        self.minTime = 5
        
        self.testSeed = 0
        self.valSeed = 1000
        self.trainSeed = np.random.randint(2000,999999999)
        
        self.period = 50
    
scheConfig = RandomConfig()
obsConfig = ObsConfig()