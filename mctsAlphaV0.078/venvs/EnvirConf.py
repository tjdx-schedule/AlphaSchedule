import numpy as np 
distribution = {
        'h' : [0.3, 20, 200],
        'm' : [0.5, 12, 125],
        'l' : [0.65, 6, 50]
        }


class EnviroConfig:
    def __init__(self):
        self.partNum = 15
        self.distType = 'h'
#------------Adjusting parameter up-------
        
        group = (self.partNum - 5) // 10
        disParam = distribution[self.distType]
        self.machNum = group * 5
        print('Resource: ',' part, mach = ', \
              self.partNum, self.machNum)
        print('Distribution Type :', self.distType, \
              ' | ', disParam)
              
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
        
class ObsConfig:
    def __init__(self, envConfig):
        w = envConfig.partNum // 5
        self.width = max(w,5)
        self.height = min(w,5)
        print('Input feature map size: ',\
              'w, h = ', self.width, self.height)
        
        self.stockFea = 5
        self.machFea = 1
        self.feaNum = self.stockFea + self.machFea
        self.stackFrameNum = 1

envConfig = EnviroConfig()
obConfig = ObsConfig(envConfig)
