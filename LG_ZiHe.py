#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 22:05:02 2018

This is the module for Logistic Regression algorithm

@author: hezi
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
import math
from sklearn.linear_model import LogisticRegression


def correlationLG_C(data1,data2,data3,data4):
    
    '''plot the relation of accuracy wrt C'''
    
    train_score=[]
    test_score=[]
    for C_val in [0.01,0.1,1,10,100]:
        clf=LogisticRegression(random_state=0,C=C_val,solver='liblinear',multi_class='ovr').fit(data1, data2.values.ravel())

        # Prediction
        test_score.append(clf.score(data3,data4))
        train_score.append(clf.score(data1,data2))
    plt.plot([-2,-1,0,1,2],train_score,color='green',label='train_score')
    plt.plot([-2,-1,0,1,2],test_score,color='red',label='test_score')
    plt.xlabel('log of C')
    plt.ylabel('Accuracy')
    plt.title('Accuracy wrt C values')
    plt.legend()
    plt.figure()
    plt.show()