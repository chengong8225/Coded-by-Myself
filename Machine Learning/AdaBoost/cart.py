# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:42:43 2019

@author: Charlie Chan
"""
# 简单的回归树
import numpy as np
import pandas as pd

class Node():
    def __init__(self, x = None):
        self.val = x
        self.left = None 
        self.right = None

def load_data():
    x = np.array(range(1,11))
    y = np.array([4.5, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.34, 8.7, 9.00]).ravel()
    return x, y

def find_best_position(x, y):   
    position = -1
    res = float("inf")
    for i in x:
        y1 = y[np.where(x<i+1e-4)]
        y2 = y[np.where(x>i+1e-4)]
        tmp = ((y1-y1.mean())**2).sum() +((y2-y2.mean())**2).sum()
        if res > tmp:
            res = tmp
            position= i
    return np.where(x == position)[0][0]
 
def regression_tree(x, y, depth = 3, nums = 2):
    tree = Node()    
    if depth == 0 or len(x) <= nums:
        tree.val = y.mean()
        return tree
    
    position = find_best_position(x, y)

    tree.val = x[position]
    tree.left = regression_tree(x[:position], y[:position], depth-1)
    tree.right = regression_tree(x[position+1:], y[position+1:], depth-1)

    return tree

def tree_predict(tree, x_pre):
    head = tree
    while head.left:
        if x_pre < head.val:
            head = head.left
        else:
            head = head.right
    return head.val

#前序遍历 循环法
def PreOrderIteration(root):
    stack = []
    sol = []
    curr = root
    while stack or curr:
        if curr:
            sol.append(curr.val)
            stack.append(curr.right)
            curr = curr.left
        else:
            curr = stack.pop()
    return sol

#中序遍历 循环法
def InOrderIteration(root):
    stack = []
    res = []
    curr = root
    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right
    return res
 

if __name__ == '__main__':
    x, y =load_data()
    retree = regression_tree(x, y, 3)
    print(InOrderIteration(retree))
    print(PreOrderIteration(retree))
    y_pre = tree_predict(retree, 4)
    print(y_pre)
    