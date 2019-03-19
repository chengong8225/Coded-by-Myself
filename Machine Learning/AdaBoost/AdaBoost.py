 # -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:39:42 2018

@author: Charlie Chan
"""

import numpy as np
import pandas as pd
import math

def GetG(D, x, y):
    ##给一个样本的权重，获得一个最好的分类器，即返回最好的v
    pos = -1
    e = 1000
    ftype = 0
    for i in range(0,max(x)):
        m1 = np.mat(G(x,i,1)!=y)*np.mat(D).T
        if m1[0,0] < e:
            pos = i
            e = m1[0,0]
            ftype = 1
        m2 = np.mat(G(x,i,2)!=y)*np.mat(D).T
        if m2[0,0] < e:
            pos = i
            e = m2[0,0]
            ftype = 2
    ##pos是当前样本权重下最好的切割点
    ##math.log((1-e)/e)/2是G的系数
    return pos, ftype, math.log((1-e)/e)/2

def G(x, pos, ftype): ##如果ftype是1，那就是>=;如果是2，那就是<=
    if ftype ==2:
        return (x<=pos)*2-1
    else:
        return (x>=pos)*2-1

def UpdateD(alpha, x, y, pos, ftype, D):
    ##这个函数是为
    m = list(map(lambda x,y,D: D * math.exp(- alpha * G(x,pos,ftype) * y), x,y,D))
    sum_m = sum(m) 
    m = list(map(lambda x: x/sum_m, m))
    return m

def CombineAllLearner(x, y, learner):  
    #learner是一个list，每个子list包含了pos, alpha, ftype
    m = []
    for i in learner:
        m.append(G(x, i[0], i[2])*i[1]) 
    k = list(2*(sum(m)>0)-1)
    return  k  ##只要最后有一个结果判断不正确 那就返回1
    
def GetFinalLearner(x, y): 
    ##这个函数是为了得出所有弱学习器
    learner  = []
    n = len(y)
    D = [1/n]*n
    while 1:
        pos, ftype, alpha = GetG(D, x, y)
        learner.append([pos, alpha, ftype])
        D = UpdateD(alpha, x, y, pos, ftype, D)
        if all(CombineAllLearner(x, y, learner)==y):
            break
    return learner

if __name__== '__main__':      
    filename = 'C:/Users/Charlie Chan/Desktop/python代码/AdaBoost/data.xlsx'
    data = pd.DataFrame(pd.read_excel(filename, header = 0)) 
    x = data['x']
    y = data['y']
    learner = GetFinalLearner(x, y)
    print(learner)