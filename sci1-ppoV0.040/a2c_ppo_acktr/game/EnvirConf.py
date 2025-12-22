import numpy as np 

distribution = {
        'h' : [0.3, 20, 200],
        'm' : [0.5, 12, 125],
        'l' : [0.65, 6, 50]
        }

class ObsConfig:
    def __init__(self, envConfig):
#        w = envConfig.partNum // 5
#        self.width = 1
#        self.height = envConfig.partNum
        self.updateWH(envConfig)
        
        self.stockFea = 9
        self.machFea = 1
        self.feaNum = self.stockFea + self.machFea
        self.stackFrameNum = 1
    
    def updateWH(self, envConfig):
        self.width = 1
        self.height = envConfig.partNum
    
    def printParam(self):
        print('Input feature map size: ',\
              'w, h = ', self.width, self.height)
        

class EnviroConfig:
    def __init__(self):
        self.partNum = 65
        self.distType = 'h'
#------------Adjusting parameter up-------
        
        group = (self.partNum - 5) // 10
        disParam = distribution[self.distType]
        self.machNum = group * 5
              
        self.tight = disParam[0]
        self.priority = disParam[1]
        
        self.maxTime = disParam[2]
        self.minTime = 5
        
        self.testSeed = 0
        self.valSeed = 1000
        self.trainSeed = np.random.randint(2000,999999999)
        
        #Enviro Rule 
        self.Kt = 1
        self.h = 1
        
    def setParam(self, partNum, machNum = None, distType = None):
        self.partNum = partNum
        if machNum is not None:
            self.machNum = machNum
        else:
            group = (self.partNum - 5) // 10
            self.machNum = group * 5
        
        if distType is not None:
            self.distType = distType
            disParam = distribution[self.distType]
            self.tight = disParam[0]
            self.priority = disParam[1]
            self.maxTime = disParam[2]
    
    def printParam(self):
        print('Resource: ',' part, mach = ', \
              self.partNum, self.machNum)
        disParam = distribution[self.distType]
        print('Distribution Type :', self.distType, \
              ' | ', disParam)
        

class Config:
    def __init__(self):
        self.envConfig = EnviroConfig()
        self.obConfig = ObsConfig(self.envConfig)
        
    def updateParam(self, partNum, machNum = None, distType = None):
        self.envConfig.setParam(partNum, machNum = machNum, distType = distType)
        self.obConfig.updateWH(self.envConfig)
        
        self.envConfig.printParam()
        self.obConfig.printParam()
        
    
config = Config()
#envConfig = EnviroConfig()
#obConfig = ObsConfig()
