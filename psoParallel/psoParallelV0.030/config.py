import numpy as np

from excel import ExcelDeal


LoadConfig = {'h':[0.3, 20, 200], 'm':[0.5, 12, 125], 'l':[0.65, 6, 50]}

class ScheConfig:
    def __init__(self):
        ex = ExcelDeal()
        
        self.partMat,self.machMat = ex.getPaLi('ft10')
        self.partNum,self.orderNum = self.partMat.shape

class RandomConfig:
    def __init__(self, load, partNum, machNum):
        self.partNum, self.machNum = partNum, machNum
        self.tight = LoadConfig[load][0]
        self.priority = LoadConfig[load][1]
        
        self.maxTime = LoadConfig[load][2]
        self.minTime = 5
        
        self.testSeed = 0
        self.valSeed = 1000
        self.trainSeed = np.random.randint(2000,10000)
        
        self.period = 50

class RanMaxConfig:
    def __init__(self):
        self.partNum, self.machNum = 20, 8
        self.tight = 0.7
        self.priority = 10
        
        self.maxTime = 100
        self.minTime = 5
        
        self.testSeed = 0
        self.valSeed = 1000
        self.trainSeed = 2000
        
        self.period = 50
        self.maxTrainSeed= self.trainSeed + self.period
        
load = 'h'        
scheConfig = RandomConfig(load, 25, 10)