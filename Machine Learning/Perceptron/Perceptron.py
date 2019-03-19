# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:51:52 2018

@author: Charlie Chan
"""

##感知机 《统计学习方法》例2.1
import numpy as np

def LoadDataSet(filename):
    X = []; Y =[]
    NumOfX = len(open(filename).readline().split('\t'))-1
    for line in open(filename).readlines():
        currx = line.strip().split('\t')
        x = []
        for i in range(NumOfX):
            x.append(float(currx[i]))
        X.append(x)
        Y.append(float(currx[-1]))
    return np.mat(X), np.mat(Y)

def UpdateCoeff(X, Y):
    n, p = np.shape(X)     #shape()获得一个矩阵的大小
    ita = 1
    w = np.zeros(p+1)      #zeros()生成一个全为0的向量
    w = np.mat(w)
    X = np.column_stack((X, np.ones(n).T)) #column_stack()为矩阵另加一列
    result = np.multiply(Y, (X*w.T).T)     #multiply()计算矩阵点乘
    while ((result<=0).any()):          
        #any(),all()类似R里的做法，不过最好事先就把对象调整到只有false，true
        for i in range(n):
            if (np.multiply(X[i,:]*w.T,Y[0,i])<=0):
                w = w + ita * Y[0, i] * X[i,:] 
                print('误判点：%d, 权重:[%f, %f, %f]'%(i+1, w[0,0], w[0,1], w[0,2]))
                result = np.multiply(Y, (X*w.T).T)
                break
    return w

if __name__ == '__main__': 
    filename = './eg01.txt'
    x, y = LoadDataSet(filename)
    weight = UpdateCoeff(x, y)
    print(weight)