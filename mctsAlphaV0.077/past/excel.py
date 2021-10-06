# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xlrd
import xlwt
from xlutils.copy import copy
import time
import datetime


class ExcelDeal:
    def __init__(self):
        sheet = pd.read_excel('jobshop.xls',skiprows=50,sheet_name= 'jobshop1')
        col = sheet.columns.values.tolist()
        new = sheet[[col[1],col[2]]]
        self.sheet = sheet
        self.col = col
        self.new = new
        
        self.nameHash = dict()
        for index,row in new.iterrows():
            if row[col[1]] == 'instance':
                self.nameHash[row[col[2]]] = index
    
    def getPaLi(self,name):
        index = self.getIndex(name)
        mat,size = self.getStaMat(index)
        mat = np.array(mat,dtype=np.int)
        paLi,machLi = self.creatPaMa(mat,size)
        return paLi,machLi
    
    def creatPaMa(self,mat,size):
        # print(mat)
        paLi = mat[:,1::2]
        machLi = mat[:,0::2]
        return paLi,machLi
       
    def getStaMat(self,index):
        sizeIndex = index+4
        staIndex = index + 5
        size = self.new.values[sizeIndex:sizeIndex+1].tolist().pop()
        mat = self.sheet.values[staIndex:staIndex+size[0],1:1+size[1]*2].tolist()
        return mat,size
#        npMat = np.array(mat,dtype = 'int')
        
    def getIndex(self,name):
        index = self.nameHash[name]
        return index
    

class ExcelLog:
    def __init__(self,name = '',isLog = False):
        self.isLog = isLog
        self.creatTime = datetime.datetime.now()
        self.name = name
        self._eval_log = -float('inf')
        if isLog:
            self.logInit()
            
    def setEval(self,evalGrade):
        if evalGrade > self._eval_log:
            self._eval_log = evalGrade
    @property   
    def getEval(self):
        return self._eval_log
            
    def setLog(self,args = None,flag = True):
        exist = getattr(self, 'excelPath', None)
        self.logInfo = args
        if flag:
            # if exist is None:
            #     self.logInit()
            # else:
            self.reset()
        self.isLog = flag
    
    def logInit(self):
        self.epoch = 0
        self.timeStr = time.strftime('%m%d%H%M',time.localtime(time.time()))
        
        self.countMax = 50000
        self.reset()
        
    def reset(self):
#        timeStr = self.timeStr#time.strftime('%m%d%H%M',time.localtime(time.time()))
        if self.name == '':
            timeStr = self.timeStr
        else:
            timeStr = self.name + '_' + self.timeStr
        self.excelPath = './models/' + timeStr + 'V'+ str(self.epoch) +'.xls' 
        self.sheetName = ['train','test','loss','info']
        self.title = [[["episode", "reward"],],\
                      [["episode", "min","average","max"],],\
                      [["episode", "kl", "value", "policy", "loss","entropy","explained_var_old","explained_var_new"],],\
                      [["key", "value"],],]
        
        self.train = []
        self.test = []
        self.loss = []
        self.infos = []

        self.creatExcel()
                
        self.count = 1
        self.testCount = 1
        self.lossCount = 1
        
        self.epoch += 1
        
        
    def creatExcel(self):
        workbook = xlwt.Workbook()  # 新建一个工作簿
        for sheet_name_index in  range(len(self.sheetName)):
            sheet_name = self.sheetName[sheet_name_index]
            sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
            value = self.title[sheet_name_index]
            index = len(value)  # 获取需要写入数据的行数
            for i in range(0, index):
                for j in range(0, len(value[i])):
                    sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
        workbook.save(self.excelPath)  # 保存工作簿
#        print("xls格式表格初始化成功！")
        
    def saveTrain(self,*args):
        if self.isLog:
            self.train.append([self.count] + list(args))
            if len(self.train) >= 5:
                self.write_excel_xls_append(0,self.train)
                self.train = []
            self.count += 1
            if self.count > self.countMax:
                self.reset()
    
    def saveTest(self,*args):
        if self.isLog:
            self.test.append([self.testCount] + list(args))
            if len(self.test) >= 2:
                self.write_excel_xls_append(1,self.test)
                self.test = []
            
            self.testCount += 1
            if self.testCount > self.countMax:
                self.reset()
    
    def saveLoss(self,*args):
        if self.isLog:
            self.loss.append([self.lossCount] + list(args))
            if len(self.loss) >= 5:
                self.write_excel_xls_append(2,self.loss)
                self.loss = []    
                
            self.lossCount += 1
            if self.lossCount > self.countMax:
                self.reset()
                
    # def saveInfo(self):
    #     if self.isLog:
    #         args = self.logInfo
    #         if args is not None:     
    #             self.infos.append(('createTime',datetime.datetime.strftime(self.creatTime,'%Y-%m-%d %H:%M:%S')))
    # #            self.infos.append(['hiddens']+hiddens)
    #             self.infos.append(['gamma',args.gamma])
    #             self.infos.append(['env_name',args.env_name])
    #             self.infos.append(['clip_param',args.clip_param])
    #             self.infos.append(['ppo_epoch',args.ppo_epoch])
    #             self.infos.append(['num_steps',args.num_steps])
    #             self.infos.append(['seed',args.seed])
    #             self.infos.append(['lr',args.lr])
    #             self.infos.append(['load',args.load])
    #             self.infos.append(['is_skip_frame',ISSKIPFRAME])
    #             self.infos.append(['skip_num',SKIPNUM])
    
    #             self.write_excel_xls_append(3,self.infos)
    #             self.infos = []
    #         print("xls info init OK！")
            
            
    def write_excel_xls_append(self,sheetIndex,value):
        path = self.excelPath
        index = len(value)  # 获取需要写入数据的行数
        workbook = xlrd.open_workbook(path)  # 打开工作簿
        sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
        worksheet = workbook.sheet_by_name(sheets[sheetIndex])  # 获取工作簿中所有表格中的的第一个表格
        rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
        new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
        new_worksheet = new_workbook.get_sheet(sheetIndex)  # 获取转化后工作簿中的第一个表格
        for i in range(0, index):
            for j in range(0, len(value[i])):
                new_worksheet.write(i+rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
        new_workbook.save(path)  # 保存工作簿
 

    
if __name__ == '__main__':
    ex = ExcelDeal()
    a,b = ex.getPaLi('ft10')