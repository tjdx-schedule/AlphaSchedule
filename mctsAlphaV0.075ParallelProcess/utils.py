# -*- coding: utf-8 -*-
import os,shutil
import numpy as np
import json


class Logger:
    def __init__(self):
        pass

    def reset(self):
        self.grade = np.float('inf')
    
    def update(self,grade):
        if grade < self.grade:
            self.grade = grade
    
    def update_arr(self,grades):
        grade = min(grades)
        if grade < self.grade:
            self.grade = grade
            
    def getBest(self):
        return self.grade

def minKIndex2(x, k, col):
    x,cou = x
    k = min(k,cou)
    
    sortIndexes = np.argsort(x.flatten())[:k]
    indices = []
    for index in sortIndexes:
        x = index // col
        y = index % col
        indices.append([x,y])
    return indices

def save_buffer(buffer, path = 'models/'):
    file_path = path + 'data2.json'
    buffer_list = []
    for state, prob, winner in buffer:
        data = []
        for state_i in state:
            data.append(state_i.tolist())
        data.append(prob.tolist())
        data.append(winner)
        buffer_list.append(data)
        # buffer_list.append((state[0].tolist(),state[1].tolist(),\
        #                     state[2].tolist(),prob.tolist(),winner))
    with open(file_path,'w',encoding='utf-8') as f:
        json.dump(buffer_list,f,ensure_ascii=False)
        
def load_buffer(buffer, path = 'models/'):
    file_path = path + 'data2.json'
    with open(file_path,'r',encoding='utf-8') as f:
        save_data = json.load(f)
    length_data = len(save_data[0])
    state_len = length_data - 2
    for one_data in save_data:
        state = []
        for i in range(state_len):
            state.append(np.array(one_data[i]))
        prob = np.array(one_data[-2])
        winner = one_data[-1]
        buffer.append((state, prob , winner))
    print('Buffer Load OK! buffer len is ',len(buffer))
    return buffer

def make_dir_init(BasePath,files=[]):
    if len(files) > 0:
        for f in files:
            try:
                os.makedirs(BasePath+f)
            except FileExistsError:
                pass
    else:
        try:
            os.makedirs(BasePath)
        except FileExistsError:
            pass
        
def mymovefile(srcfile, dstfile):
    '''
    move file to another dirs
    '''
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile, dstfile)
        # print("move %s -> %s" % (srcfile, dstfile))

def mycpfile(srcfile, dstfile):
    '''
    copy file to another dirs
    '''
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.copy(srcfile, dstfile)
        # print("move %s -> %s" % (srcfile, dstfile))