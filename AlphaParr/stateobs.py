import math
import numpy as np

class StateObserve:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        # self.machDictInit()
        
        self.partNum = self.scheduler.genpart.partNum
        # self.orderNum = self.scheduler.genpart.orderNum
        self.machNum = self.scheduler.genpart.machNum
        
        self._h = 1
        self._Kt = 1
                   
    def observation(self):
        obs = []
        
        partFea = np.array(self.partEncode())
        machFea = np.array(self.machEncode())
               
        obs = [partFea,machFea]
        return obs
    
    def partEncode(self):
        e = 1e-10
        
        partFeature = []
        cur = np.min(self.scheduler.mach)
        for partIndex in range(self.partNum):
              hours, deadline, priority = self.scheduler.part[partIndex]
              deadline_cur = (deadline - cur)*(deadline > 0)
              deadline_cur_hours = deadline_cur - hours
              priority_hours = priority/(hours+e)
             
              slack = -(deadline_cur_hours)/(hours+e)
              wspt = hours/(priority+e)
              wmdd = -max(hours,deadline_cur)/(priority+e)
              atc = math.exp(-max(deadline_cur_hours,0)/(self._h*hours+e))*priority_hours
              wcovert = max(1-max(deadline_cur_hours,0)/(self._Kt*hours+e),0)*priority_hours
             
              feature = [hours, priority,
                        deadline_cur,deadline_cur_hours,
                        slack,wspt,wmdd,atc,wcovert]
              partFeature.extend(feature)
        
        machTime = (self.scheduler.mach - cur).tolist()
        partFeature.extend(machTime)
        return partFeature
            
            

    def machEncode(self):
        grade = self.scheduler.getGrade()
        return grade
            # machPartIndex = 
    

class VecNorm:
    def __init__(self):
        from config import  scheConfig
        self.partNum = scheConfig.partNum
        self.machNum = scheConfig.machNum
        
        self.part_ob_rms = RunningMeanStd(shape=(8*self.partNum+self.machNum))
        # self.mach_ob_rms = RunningMeanStd(shape=(2*self.machNum,self.partNum,4))
        
        self.clipob = 5
        self.epsilon = 1e-8
        
        self.max_norm = -float('inf')
        self.min_norm = float('inf')   
        self.clipre = 1
    
    def ob_norm(self, obs):
        partFea,machFea = obs
        partFea = self._obfilt(partFea,'part')
        machFea = self._refilt(machFea)
        # machFea = self._obfilt(machFea,'mach')
        # partFea = np.expand_dims(self._obfilt(partFea,'part'),0)
        # machFea = np.expand_dims(self._obfilt(machFea,'mach'),0)
        return [partFea,machFea]
    
    def _obfilt(self, obs, scope):
        if scope == 'part':
            ob_rms = self.part_ob_rms
        elif scope == 'mach':
            ob_rms = self.mach_ob_rms
        ob_rms.update(obs)
        obs = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + self.epsilon), -self.clipob, self.clipob)
        return obs
    
    def _refilt(self, grade):
        grade = -grade
        self.min_norm = min(grade,self.min_norm)
        self.max_norm = max(grade,self.max_norm)
        
        half = (self.max_norm + self.min_norm)/2
        delta = (self.max_norm - self.min_norm)/2
        
        grade = (grade - half)/(delta+1e-10)
        grade = np.clip(grade,-self.clipre,self.clipre)
        return grade  

    
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