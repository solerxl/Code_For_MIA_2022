# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 00:26:26 2021

@author: 11075
"""
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
import keras.backend as K
# 回调函数
class LearningHandler(Callback):
    '''
    Class for managing the learning rate scheduling and early stopping criteria

    Learning rate scheduling is implemented by multiplying the learning rate
    by 'drop' everytime the validation loss does not see any improvement
    for 'patience' training steps
    '''
    def __init__(self, lr, drop, lr_tensor, patience):
        '''
        lr:         initial learning rate
        drop:       factor by which learning rate is reduced by the
                    learning rate scheduler
        lr_tensor:  tensorflow (or keras) tensor for the learning rate
        patience:   patience of the learning rate scheduler
        '''
        super(LearningHandler, self).__init__()
        self.lr = lr
        self.drop = drop
        self.lr_tensor = lr_tensor
        self.patience = patience

    def on_train_begin(self, logs=None):
        '''
        Initialize the parameters at the start of training (this is so that
        the class may be reused for multiple training runs)
        '''
        self.assign_op = tf.no_op()
        self.scheduler_stage = 0
        self.best_loss = np.inf
        self.wait = 0
        self.loss_former_1 = 1e-8
        self.loss_former_2 = 1e-8
        self.loss_former_3 = 1e-8

    def on_epoch_end(self, epoch, logs=None):
        '''
        Per epoch logic for managing learning rate and early stopping
        '''
        stop_training = False
        # check if we need to stop or increase scheduler stage
        if isinstance(logs, dict):
            loss = logs['loss']
        else:
            loss = logs
        if loss <= self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.scheduler_stage += 1
                self.wait = 0

        # calculate and set learning rate
        lr = self.lr * np.power(self.drop, self.scheduler_stage)
        K.set_value(self.lr_tensor, lr)
        
#         tf.summary.scalar('learning_rate',self.lr)
        
        # built in stopping if lr is way too small
        if lr <= 1e-8 or (epoch>=5 and abs(logs['loss']-self.loss_former_3)/self.loss_former_3<=0.001 and abs(self.loss_former_3-self.loss_former_2)/self.loss_former_2<=0.001 and abs(self.loss_former_2-self.loss_former_1)/self.loss_former_1<=0.001):
            stop_training = True

        # for keras
        if hasattr(self, 'model') and self.model is not None:
            self.model.stop_training = stop_training
            
        self.loss_former_1 = self.loss_former_2
        self.loss_former_2 = self.loss_former_3
        self.loss_former_3 = logs['loss']
        return stop_training