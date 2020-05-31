# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


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
    
if __name__ == '__main__':
    ex = ExcelDeal()
    a,b = ex.getPaLi('ft10')