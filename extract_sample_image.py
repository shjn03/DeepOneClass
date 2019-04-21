#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:43:33 2019

@author: shingo
"""

from keras.datasets import mnist,cifar10
import cv2
import os
num_classes=10
datasets=["cifar10"]
outbasedir="data"
for dataname in datasets:
    #train
    if dataname=="mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        class_names=[str(i)for i in range(10)]
    elif dataname=="cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        class_names=[
                        'airplane',
                        'automobile',
                        'bird',
                        'cat',
                        'deer',
                        'dog',
                        'frog',
                        'horse',
                        'ship',
                        'truck']    
    
    
    os.makedirs(os.path.join(outbasedir,dataname),exist_ok=True)
    for class_name in class_names:
        os.makedirs(os.path.join(outbasedir,dataname,"train",class_name),exist_ok=True)
        os.makedirs(os.path.join(outbasedir,dataname,"test",class_name),exist_ok=True)
        
    for idx in range(len(y_train)):
        class_name=class_names[y_train[idx][0]]
        img=X_train[idx]
        cv2.imwrite(os.path.join(outbasedir,dataname,"train",str(class_name),class_name+"_"+str(idx)+".png"),img)
    for idx in range(len(y_test)):
        class_name=class_names[y_test[idx][0]]
        img=X_test[idx]
        cv2.imwrite(os.path.join(outbasedir,dataname,"test",str(class_name),class_name+"_"+str(idx)+".png"),img)
        
        
        
    