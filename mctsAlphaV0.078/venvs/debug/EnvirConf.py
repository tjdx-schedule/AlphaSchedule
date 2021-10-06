import numpy as np 

class EnviroConfig:
    def __init__(self):
        self.partNum, self.machNum =55,25
        self.tight = 0.65
        self.priority = 14
        
        self.maxTime = 150
        self.minTime = 5
        
        self.testSeed = 0
        self.valSeed = 1000
        self.trainSeed = np.random.randint(2000,999999999)
        
        #Enviro Rule 
        self.Kt = 1
        self.h = 1
        
class ObsConfig:
    def __init__(self):
        self.width = 11
        self.height = 5
        
        self.stockFea = 5
        self.machFea = 1
        self.feaNum = self.stockFea + self.machFea
        self.stackFrameNum = 1

envConfig = EnviroConfig()
obConfig = ObsConfig()
