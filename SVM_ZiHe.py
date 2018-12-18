#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:42:31 2018

This is the module for SVM algorithm

@author: hezi
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
import math



def datasplit(number_training_examples=5000):
    
    '''break the data into training set and test set'''

    labeled_images = pd.read_csv('train.csv')
    images = labeled_images.iloc[0:number_training_examples,1:]
    labels = labeled_images.iloc[0:number_training_examples,:1]
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
    return train_images, test_images,train_labels, test_labels


def view_one_image(data1, data2, i=34):
    
    '''plot one example of digit'''
    
    img=data1.iloc[i].as_matrix()
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title('digit {}'.format(data2.iloc[i,0]))
    plt.show()


def feature_scaling(data1,data2,denominator=255):
    '''feature scaling'''
    test_images=data1.divide(denominator)
    train_images=data2.divide(denominator)
    return test_images, train_images

def histogram(data1,i=1):
    
    '''plot distribution of pixel values'''
    
    plt.hist(data1.iloc[i])
    plt.title('histogram of pixel values')
    plt.figure()
    plt.show()



def correlationSVM_C_gamma(data1,data2,data3,data4):
    
    '''plot the relation of accuracy wrt c and gamma'''
    
    CC=['blue','red','orange','green','black']
    LL=['C=0.1','C=1','C=10','C=100','C=1000']
    index=0
    for c in [0.1,1,10,100,1000]:
        list_A=[]
    
        for g in [0.0001,0.001,0.01,0.05,0.1]:
        
            clf = svm.SVC(C=c,gamma=g)
            clf.fit(data1, data2.values.ravel())
            list_A.append(clf.score(data3,data4))
    
        plt.plot([-4,-3,-2,math.log10(0.05),-1],list_A,color=CC[index],label=LL[index])
        index=index+1
    plt.xlabel('log of gamma')
    plt.ylabel('Accuracy')
    plt.title('Accuracy wrt gamma and C values')
    plt.legend()
    plt.figure()
    plt.show()
    
def plot_Gaussian_small_sigma():
    
    '''plot Gaussian pdf for small sigma value'''
    
    x = np.linspace(-10, 10, 101)
    P_norm = norm.pdf(x, 0, 1)
    plt.plot(x, P_norm, 'g', linewidth=1.0, label = "variance=1")
    plt.plot(3,norm.pdf(3, 0, 1),marker='x',color='r',label='x2=3')
    plt.plot(-1,norm.pdf(-1, 0, 1),marker='x',color='orange',label='x1=-2')
    plt.plot(0,norm.pdf(0, 0, 1),marker='*',color='black',label='mean point')
    plt.legend()
    plt.title('Gaussian PDF for small value of sigma')
    plt.figure()
    plt.show()
    
    
def plot_Gaussian_large_sigma():
    
    '''plot Gaussian pdf for large sigma value'''
    
    x = np.linspace(-10, 10, 101)
    P_norm = norm.pdf(x, 0, 100)
    plt.plot(x, P_norm, 'b', linewidth=1.0, label = "variance=100")
    plt.plot(3,norm.pdf(3, 0, 100),marker='x',color='r',label='x2=3')
    plt.plot(-1,norm.pdf(0, 0, 100),marker='x',color='orange',label='x1=-2')
    plt.plot(0,norm.pdf(0, 0, 100),marker='*',color='black',label='mean point')
    plt.legend()
    plt.title('Gaussian PDF for large value of sigma')
    plt.figure()
    plt.show()