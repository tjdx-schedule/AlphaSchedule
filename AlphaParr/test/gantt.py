# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def gatt(m, job, t):
	"""甘特图
	m机器集
	job工序顺序集
	t时间集
	"""
	for j in range(len(m)):  # 工序j
		i = m[j] - 1  # 机器编号i
		if j == 0:
			plt.barh(i, t[j])
			plt.text(np.sum(t[:j + 1]) / 8, i, 'J%s\nT%s' % ((job[j]), t[j]), color="white", size=10)
		else:
			plt.barh(i, t[j], left=(np.sum(t[:j])))
			plt.text(np.sum(t[:j]) + t[j] / 8, i, 'J%s\nT%s' % ((job[j]), t[j]), color="white", size=10)
	plt.yticks(np.arange(max(m)), np.arange(1, max(m) + 1))



m = np.random.randint(1, 7, 15)  # 生成工序所在机器编号
job = np.arange(1, 16)  # 生成工序编号
np.random.shuffle(job)
t = np.random.randint(18, 25, 15)  # 生成工序时间
gatt(m, job, t)
plt.savefig("test.png", dpi=1200)
plt.show()
