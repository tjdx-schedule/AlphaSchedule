import matplotlib.pyplot as plt
import random


class TreeLogger:
    def __init__(self):
       self.tree = {'root':{}}
       self.node = self.tree['root']
       self.layer = 1
       
    def grow(self, node, lastAct):
        if self.layer <= 3:
            for key, value in node._children.items():
                if lastAct != key:
                    self.node[key] = self._readValue(value)
                else:
                    tempKey = self._readValue(value)
                    self.node[key] = {tempKey:{}}
    
            self.node = self.node[lastAct][tempKey]
            self.layer += 1
        
       
    def _readValue(self, node):
        statics = ''
        statics += str(round(node._n_visits,3)) #+ '-'
        # statics += str(round(node._P,3)) + '-'
        # statics += str(round(node._Q,3))
        return statics
    
    def _outputDict(self):
        return self.tree
    
    def plotTree(self):
        createPlot(self.tree)

def plotGantt(partList,machineNum):
    plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签  
    plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号  
    
    height=16 # 柱体高度，设为2的整数倍，方便Y轴label居中，如果设的过大，柱体间的间距就看不到了，需要修改下面间隔为更大的值
    interval=4 # 柱体间的间隔
    colors = ("red","yellow","green","brown","blue") # 颜色，不够再加
    x_label=u"调度时刻" # 设置x轴label
    
    partLen, _ = partList.shape
    labels= ['M'+str(i) for i in range(machineNum)]
    fig,ax=plt.subplots(figsize=(10,5))
    count=0;
    
    for i in range(partLen):
        color = randomcolor()#colors[i]
        partArr = partList[i]

        index = partArr[2]
        timeTuple = (partArr[0], partArr[1] - partArr[0])
        ax.broken_barh([timeTuple],\
                    ((height+interval)*index+interval,height), \
                    facecolors= color)
        operaStr = str(i) + "-" + str(0)
        plt.text(partArr[0], (height+interval)*(index+0.5),\
                 operaStr,fontsize='large') 
        
        count = max(count,partArr[1])
       
    ax.set_ylim(0, (height+interval)*len(labels)+interval)
    ax.set_xlim(0, count+2)
    ax.set_xlabel(x_label)
    ax.set_yticks(range(int(interval+height/2),int((height+interval)*len(labels)),\
                        int((height+interval))))
    ax.set_yticklabels(labels)
    
    plt.show()
    
    
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
 

#获取叶节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    firstSide = list(myTree.keys())
    firstStr = firstSide[0]
    secondDict = myTree[firstStr]
    for key in secondDict:
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs
def getTreeDepth(myTree):
    maxDepth = 0
    firstSide = list(myTree.keys())
    firstStr = firstSide[0]
    secondDict = myTree[firstStr]
    for key in secondDict:
        if type(secondDict[key]).__name__ =='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth
 
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
 
 
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
 
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstSide = list(myTree.keys())
    firstStr = firstSide[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
 
 
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()