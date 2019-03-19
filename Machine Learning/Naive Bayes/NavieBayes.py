# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:47:24 2018

@author: Charlie Chan
"""

import pandas as pd
import numpy as np
from functools import reduce

def NaiveBayes(data, item):
    NumOfX = data.columns.size - 1
    NumOfRow = data.ix[:,0].size
    state = set(data.ix[:,-1])
    ConditionP = [0]*NumOfX
    P = [0]*NumOfX
    label = []
    FinalP = []  
    for i in state:
        P0 = sum(data.ix[:,-1]==i)/NumOfRow
        label.append(i)
        for j in range(NumOfX):
            P[j] = sum(data.ix[:,j]==item[j])/NumOfRow
            ConditionP[j] = sum(data[data.ix[:,-1]==i].ix[:,j]==item[j])/NumOfRow
        p1 = reduce(lambda x,y :x * y, P)
        p2 = reduce(lambda x,y :x * y, ConditionP)
        FinalP.append(p2*P0/p1)      
    return label[FinalP.index(max(FinalP))]

filename = 'C:/Users/Charlie Chan/Desktop/python代码/朴素贝叶斯/data.xlsx'
data = pd.DataFrame(pd.read_excel(filename, header=0)) ##header=0代表读取第一行做列名
item = ['帅','好','高','上进']
NaiveBayes(data, item)