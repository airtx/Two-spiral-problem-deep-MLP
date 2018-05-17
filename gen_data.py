# -*- coding: utf-8 -*-
"""
Created on Thu May  3 17:54:18 2018

@author: rafael

Generate and plot two spiral problem data
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## size of the dataset
data_size = 100

th = np.linspace(0,20,data_size)
## 1st spiral
x1, y1 = [th/4*np.cos(th), th/4*np.sin(th)]

## 2nd spiral
x2, y2 = [(th/4 + 0.8)*np.cos(th), (th/4 + 0.8)*np.sin(th)]

## generating training data
train_data = np.concatenate((np.array([x1,y1]),np.array([x2,y2])), axis = 1).T
train_labels = np.concatenate((-1*np.ones(data_size), np.ones(data_size))).reshape((2*data_size, 1))

#########################################################################################
sns.set()
sns.set_style("white")
fig = plt.figure()
plt.scatter(x1,y1, marker = 'x', color = 'green', s = 40, linewidth = 2, label = '$C_1$')
plt.scatter(x2,y2, marker = 'o', color = 'blue', s = 40, label = '$C_2$')

legend = plt.legend(loc = 'upper left', fontsize = 'large', frameon=True, shadow = True)
#legend.get_frame().set_facecolor('#ffffff')
plt.axis([-6,6,-6,6])

plt.xlabel('$x$', fontsize = 'large')
plt.ylabel('$y$', fontsize = 'large')
plt.title('Problema das duas espirais', fontsize = 'large')

plt.show()

#########################################################################################

## test data size
test_size = int(data_size/2)

th = np.linspace(0,20,2*data_size)
th1 = np.random.choice(th, size = test_size)
th2 = np.random.choice(th, size = test_size)
## 1st spiral
x1, y1 = [th1/4*np.cos(th1), th1/4*np.sin(th1)]

## 2nd spiral
x2, y2 = [(th2/4 + 0.8)*np.cos(th2), (th2/4 + 0.8)*np.sin(th2)]

## validaton data
test_data = np.concatenate((np.array([x1,y1]),np.array([x2,y2])), axis = 1).T
test_labels = np.concatenate((-1*np.ones(test_size), np.ones(test_size))).reshape((2*test_size, 1))

## saving data
np.savez('data.npz', train_data = train_data, train_labels = train_labels,
         test_data = test_data, test_labels = test_labels)


