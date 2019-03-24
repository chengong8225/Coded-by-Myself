# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:42:57 2019

@author: Charlie Chan
"""

# GBDT
#《统计学习方法》 P149 例8.2
import numpy as np

def load_data():
    x = np.array(range(1,11))
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.7, 9.00, 9.05]).ravel()
    return x, y

def find_best_position(x, y):   
    position = -1
    m = []
    new_x = [x[i]/2+x[i-1]/2 for i in range(1, len(x))]
    for i in new_x:
        y1 = y[np.where(x<i)]
        y2 = y[np.where(x>i)]
        tmp = ((y1-y1.mean())**2).sum() +((y2-y2.mean())**2).sum()
        m.append(tmp)
    position = np.where(m == min(m))[0][0]
    y1 = y[np.where(x<new_x[position])]
    y2 = y[np.where(x>new_x[position])]
    pre = [y1.mean()]*len(y1)+[y2.mean()]*len(y2)
    residual = y - np.array(pre)
    return new_x[position], residual, y1.mean(), y2.mean()

def get_learners(x, y, nums=5, max_loss=0.1):
    res = []
    for i in range(nums):
        split_point, residual, left, right = find_best_position(x, y)
        res.append((split_point, round(left,2), round(right,2)))
        y = residual
        if (residual**2).sum() <= max_loss:
            break
    return res

def gbdt_predict(learners, x_pre):
    y_pre = 0
    for i in learners:
        y_pre += i[2] if x_pre > i[0] else i[1]
    return round(y_pre, 2)

if __name__ == '__main__':
    x, y =load_data()
    res = get_learners(x, y, nums = 6)
    print(res)
    print(gbdt_predict(res, 5))
