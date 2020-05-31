# -*- coding: utf-8 -*-
import numpy as np
import json


a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2,3],[4,5,6]])

d = [[a,1],[b,2]]

data_write = []
for i in d:
    data_write.append((i[0].tolist(),i[1]))

# 写入数据到文件
with open('1/data2.json','w',encoding='utf-8') as f:
  json.dump(data_write,f,ensure_ascii=False)
# 从文件读取数据
# with open('data2.json','r',encoding='utf-8') as f:
#     data = json.load(f)
# print(data)


