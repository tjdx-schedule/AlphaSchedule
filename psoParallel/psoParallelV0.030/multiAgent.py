import numpy as np
from pso import ParticleSwarmOptimization
#experiment arguments
c1 = 2
c2 =2.1
Wstart = 0.9
Wend = 0.4
popuSize = 100 
subPopNum = 2
subPopSize = np.int(popuSize/subPopNum)
iterationTimes = 100
class Agent:
    def __init__(self,c1,c2,Wstart, Wend, popuSize, scheduler):
        self.swarm = ParticleSwarmOptimization(c1, c2, Wstart, Wend, popuSize, scheduler)
    
    def runPSO(self, iterationTimes):
        self.swarm.iteration(iterationTimes)

    def sendSwarm(self, targetAgent):
        targetAgent.swarm.population = self.swarm.population

    # def dividePop(self, targetAgents):
    #     agentNum = len(targetAgents)
    #     subPopSize = np.int(popuSize/agentNum)
    #     i = 0
    #     for i in (0, agentNum-1):
    #         targetAgents[i].swarm.population = self.swarm.population[i*subPopSize:(i+1)*subPopSize]
    #     targetAgents[i+1] = self.swarm.population[(i+1)*subPopSize:]

    def dividePop(self, targetAgent1, targetAgent2):
        targetAgent1.swarm.population = self.swarm.population[0:subPopSize]
        targetAgent2.swarm.population = self.swarm.population[subPopSize:] 

    def migrate(self, targetAgent1, targetAgent2):
        targetAgent2.swarm.population = targetAgent1.swarm.PsoOp.indiBestAnswer
        targetAgent1.swarm.population = targetAgent1.swarm.PsoOp.indiBestAnswer
         
    def receiveBest(self, sourceAgent1, sourceAgent2):
        self.swarm.population[0:subPopSize] = sourceAgent1.swarm.population
        self.swarm.population[subPopSize:] = sourceAgent2.swarm.population

if __name__ == '__main__':
    BA = Agent(c1, c2, Wstart, Wend, popuSize)
    SA = Agent(c1, c2, Wstart, Wend, popuSize)
    EA1 = Agent(c1, c2, Wstart, Wend, subPopSize)
    EA2 = Agent(c1, c2, Wstart, Wend, subPopSize)
    BA.swarm.PsoOp.sche.recreate('test')
    EA1.swarm.PsoOp.sche.recreate('test')
    EA2.swarm.PsoOp.sche.recreate('test')
    BA.swarm.initPopulation()
    BA.sendSwarm(SA)
    SA.dividePop(EA1, EA2)
    EA1.runPSO()
    EA2.runPSO()
    SA.migrate(EA1,EA2)
    BA.receiveBest(EA1, EA2)
    BA.runPSO()
    print(BA.swarm.PsoOp.globalBestGrade)
    print(BA.swarm.PsoOp.globalBestIndi)