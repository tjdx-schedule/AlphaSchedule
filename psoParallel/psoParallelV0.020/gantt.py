import matplotlib.pyplot as plt
import random

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
