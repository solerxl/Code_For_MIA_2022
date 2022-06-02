 # -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 23:57:33 2021

@author: 11075
"""
import tensorflow as tf
import pandas as pd
import keras.backend as K
import numpy as np

def new_MSE(y_true, y_pred):
    
    # 添加mask
    mask = tf.logical_not(tf.equal(y_true, -5))
    mask = tf.cast(mask, dtype=y_true.dtype)
    # 计算MSE
    y_true_masked = mask*y_true
    y_pred_masked = mask*y_pred
    a1 = K.sum(K.square(y_pred_masked -y_true_masked))/(K.sum(mask)+1e-16)
    return a1



def MAE(y_true, y_pred):
#    y_true = K.print_tensor(y_true,message = 'y_true = ')
#    y_pred = K.print_tensor(y_pred,message = 'y_pred = ')
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    mask = tf.logical_not(tf.equal(y_true, -5.0))
    mask = tf.cast(mask, dtype=y_true.dtype)
    y_true *=  mask
    y_pred *=  mask
    res = K.sum(K.abs(y_true - y_pred))/(K.sum(mask)+1e-16)
    return res
