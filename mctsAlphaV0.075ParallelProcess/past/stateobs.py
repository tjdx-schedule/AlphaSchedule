import numpy as np

from config import obsConfig

class StateObserve:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        # self.machDictInit()
        
        self.partNum = self.scheduler.genpart.partNum
        self.machNum = self.scheduler.genpart.machNum
        
        self.fea_num = obsConfig.ruleFeatureNum
        self.stackN = obsConfig.stackFrameNum
        
        self._h = 1
        self._Kt = 1
    
    def pastStateInit(self):
        self.stackPartFeature = np.zeros((self.stackN*self.partNum,self.fea_num))
        self.stackMachFeature = np.zeros((self.stackN,self.machNum))
        
    def stackPastFea(self,partFea,machTime):
        if self.stackN > 1:
            self.stackPartFeature[:-self.partNum] = self.stackPartFeature[self.partNum:]
            self.stackMachFeature[:-1] = self.stackMachFeature[1:]
        self.stackPartFeature[-self.partNum:] = partFea
        self.stackMachFeature[-1] = machTime
        return self.stackPartFeature.copy(),self.stackMachFeature.copy()
        
    def observation(self):
        obs = []
        
        partFea,machTime = np.array(self.partEncode())
        machFea = np.array(self.machEncode())
        
        stackPartFea,stackMachTime = self.stackPastFea(partFea,machTime)
        obs = [stackPartFea,stackMachTime,machFea]
        return obs
 
        
    def partEncode(self):
        e = 1e-10
        cur = np.min(self.scheduler.mach)
        
        hours, deadline, priority = np.hsplit(self.scheduler.part,3)
        
        deadline_cur = (deadline - cur)*(deadline > 0)
        deadline_cur_hours = deadline_cur - hours

        
        slack = -(deadline_cur_hours)/(hours+e)

        partFeatureMat = np.concatenate((hours, priority,
                        deadline_cur,deadline_cur_hours,
                        slack),axis=1)       
        partFeature = partFeatureMat
        machTime = self.scheduler.mach - cur
        
        # feature = np.hstack((partFeature,machTime))
        return [partFeature,machTime]
            
    def machEncode(self):
        grade = self.scheduler.getGrade()
        return grade
            # machPartIndex = 
    

class VecNorm:
    def __init__(self,ret_rms = None):
        from config import  scheConfig
        self.partNum = scheConfig.partNum
        self.machNum = scheConfig.machNum    
        
        self.fea_num = obsConfig.ruleFeatureNum
        self.stackN = obsConfig.stackFrameNum
        
        self.partMatWidth = obsConfig.width
        self.partMatHeight = obsConfig.height
        self.partPlaneNum = self.stackN*self.fea_num
        self.wh = self.partMatWidth * self.partMatHeight
        
        self.clipob = 5
        self.epsilon = 1e-8
        self.ret_rms = RunningMeanStd(shape=()) if ret_rms is None \
            else ret_rms
        
        self.max_norm = -float('inf')
        self.min_norm = float('inf')   
        self.clipre = 1

    def ob_norm(self, obs):
        partFea,machTime,machFea = obs
        # machFea = self._refilt(machFea)
        
        feature_stacked = self.stackPastFeature(partFea,machTime)
        return [feature_stacked,machFea]
    
    def inverse_rew(self, ret):
        inver_ret = -ret * np.sqrt(self.ret_rms.var + self.epsilon)
        return inver_ret
        
    def _refilt(self, grade):
        grade = -grade
        self.min_norm = min(grade,self.min_norm)
        self.max_norm = max(grade,self.max_norm)
        
        half = (self.max_norm + self.min_norm)/2
        delta = (self.max_norm - self.min_norm)/2
        
        grade = (grade - half)/(delta+1e-10)
        grade = np.clip(grade,-self.clipre,self.clipre)
        return grade  

        
    def stackPastFeature(self, partFea, machTime):
        partFea_N = np.hsplit(partFea.T,self.stackN) #N 5*15
        partFea_3d = np.array(partFea_N)#N*5*15
        partFea_3d_reshape = partFea_3d.reshape(self.partPlaneNum,self.partMatWidth,self.partMatHeight)
        
        machTime_mat = np.zeros((self.stackN,self.wh))
        machTime_mat[:,:self.machNum] = machTime
        machTime_reshape = machTime_mat.reshape(self.stackN,self.partMatWidth,self.partMatHeight)
        frame = np.concatenate((partFea_3d_reshape,machTime_reshape),axis=0)
        return frame
                      
    
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count    