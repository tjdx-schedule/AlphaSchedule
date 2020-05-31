# -*- coding: utf-8 -*-
import numpy as np
import json
from collections import deque

def save_buffer(buffer):
    buffer_list = []
    for state, prob, winner in buffer:
        buffer_list.append((state[0].tolist(),state[1].tolist(),prob.tolist(),winner))
    with open('models/data2.json','w',encoding='utf-8') as f:
        json.dump(buffer_list,f,ensure_ascii=False)
        
def load_buffer(buffer):
    with open('models/data2.json','r',encoding='utf-8') as f:
        data = json.load(f)
    for state0, state1, prob, winner in data:
        state0 = np.array(state0)
        state1 = np.array(state1)
        state = [state0,state1]
        prob = np.array(prob)
        buffer.append((state, prob , winner))
    print('Buffer Load OK! buffer len is ',len(buffer))
    return buffer

