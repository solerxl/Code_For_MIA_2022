#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from re import X
import sys
sys.path.append('./')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 屏蔽log信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from tqdm import tqdm
import scipy.io as sio
import numpy as np
import tensorflow as tf
import keras.backend as K
import datetime
from sklearn.model_selection import KFold
from keras.callbacks import TensorBoard
# 去除不必要的警告信息
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
# automatically tuning the parameters
from hyperopt import fmin, hp,rand

# 导入模型需要的损失，指标等
import utils.model_metric as met
import utils.model_callback as cbk


# In[ ]:
import keras
from keras import Input,Model
# from keras.layers import
from keras.layers import Dense,Lambda,Add,LSTM,GRU
from keras.engine.topology import Layer
H_temp = 0

# 隐表示层
class LatentLayer(Layer):
    def __init__(self, train_len ,l_dim,**kwargs):
        super(LatentLayer, self).__init__(**kwargs)
        self.train_len = train_len
        self.dim = l_dim
    def build(self, input_shape):
        self.H = self.add_weight(name='H',
                                 shape=(self.train_len,input_shape[1],self.dim),
                                 initializer='glorot_normal',
                                 trainable=True)
        super(LatentLayer, self).build(input_shape)
    def call(self, x):
        return self.H
    def compute_output_shape(self, input_shape):
        return (self.train_len,input_shape[1],self.dim)

# 扩展维度
def expand_dimention(x, ind):
    return tf.expand_dims(tf.cast(x,'float32'),ind)

# tensor切片
def tensor_slice(x,ax,ind):
    return tf.gather(x,axis = ax,indices = [ind])

# In[ ]: Missihng Imputation Strategies:

# impute missing score
class HardCalculateLayer_1(Layer):
    def __init__(self, **kwargs):
        super(HardCalculateLayer_1, self).__init__(**kwargs)
    def build(self, input_shape):
        super(HardCalculateLayer_1, self).build(input_shape)
    def call(self, x):
        input_predict,input_true = x
        # 填充之前模态完全缺失的时刻
        true_input_gate_0 = tf.logical_not(tf.equal(input_true, -5))    # 输入空为0，有为1
        pred_input_gate_0 = tf.equal(input_true, -5)                    # 输入空为1，有为0
        true_input_gate_0 = tf.cast(true_input_gate_0, dtype=input_true.dtype)
        pred_input_gate_0 = tf.cast(pred_input_gate_0, dtype=input_true.dtype)
        temp1 = pred_input_gate_0*input_predict
        temp2 = true_input_gate_0*input_true
        input_temp = Add()([temp1,temp2])
        return input_temp

# missing mask
class HardCalculateLayer_2(Layer):
    def __init__(self, **kwargs):
        super(HardCalculateLayer_2, self).__init__(**kwargs)
    def build(self, input_shape):
        super(HardCalculateLayer_2, self).build(input_shape)
    def call(self, x):
        input_predict,input_true,true_input_gate = x
        # 填充之前模态完全缺失的时刻
        true_input_gate = tf.tile(true_input_gate,[1,input_predict.shape[-1]])
        true_input_gate = tf.expand_dims(true_input_gate,1)
        pred_input_gate = 1-true_input_gate                    # 输入空为1，有为0
        temp1 = pred_input_gate*input_predict
        temp2 = true_input_gate*input_true
        input_temp = Add()([temp1,temp2])
        return input_temp


class HardCalculateLayer_3(Layer):
    def __init__(self, **kwargs):
        super(HardCalculateLayer_3, self).__init__(**kwargs)
    def build(self, input_shape):
        super(HardCalculateLayer_3, self).build(input_shape)
    def call(self, x):
        input_predict,input_true,true_input_gate = x
        # 首先标记之前模态完全缺失的时刻
        true_input_gate_1 = tf.tile(true_input_gate,[1,input_predict.shape[-1]-4]) # 输入完整为1，缺失为0
        true_input_gate_1 = Lambda(lambda x: K.concatenate(x, axis=1))([true_input_gate_1,tf.ones([true_input_gate.shape[0],4],dtype=true_input_gate_1.dtype)])  # 评分部分不需要补
        true_input_gate_1 = tf.expand_dims(true_input_gate_1,1)
        pred_input_gate_1 = 1-true_input_gate_1                    # 输入缺失为1，有为0
        pred_input_gate_1 = tf.cast(pred_input_gate_1, dtype=input_true.dtype)
        input_true = Add()([-5*pred_input_gate_1,input_true*true_input_gate_1])   # 把对应缺失时刻变成-5

        # 接着标记缺失的评分
        pred_input_gate = tf.equal(input_true, -5)                    # 输入空为1，有为0
        pred_input_gate = tf.cast(pred_input_gate, dtype=input_true.dtype)
        true_input_gate = 1-pred_input_gate                        # 输入有为1，空为0

        temp1 = pred_input_gate*input_predict
        temp2 = true_input_gate*input_true
        input_temp = Add()([temp1,temp2])
        return input_temp

# In[ ]:
def define_training_model(features_in,features_out,latent_dim,seq_num,pred_length,train_num,rnn_unit,view_num,act_fun):
    mid_layer_num = 1
    model_list ={'LSTM':LSTM(latent_dim,return_state = True,return_sequences = True, name = 'encoder_LSTM'),
                 'GRU':GRU(latent_dim,return_state = True,return_sequences = True, name = 'encoder_GRU')}

    with tf.variable_scope('Input'):
        Input_1 = Input(batch_shape = (train_num,seq_num,features_in[2]))  # Demographics

    with tf.variable_scope('FusionH'):
        encoder_inputs = LatentLayer(train_len = train_num,l_dim= latent_dim,name = 'LatentLayer_train')(Input_1)   # shape: n x seq x latent_dim

    with tf.variable_scope('FusionDegration'):
        De_layer = []      # Degradation layers
        for i in range(view_num):
            temp_de_layer = []
            for j in range(mid_layer_num):
                temp_de_layer.append(Dense(latent_dim, name = 'Degeneration_Layer_view'+str(i+1)+'_'+str(j), activation = act_fun))
            temp_de_layer.append(Dense(features_in[i], name = 'Degeneration_Layer_view'+str(i+1)+'_output', activation = act_fun))
            De_layer.append(temp_de_layer)
        Recon_X = []
        for i in range(view_num):
            temp_input = De_layer[i][0](encoder_inputs)
            for j in range(1,mid_layer_num):
                temp_input = De_layer[i][j](temp_input)
            Recon_X.append(De_layer[i][-1](temp_input))  # shape: n x seq x dv



    with tf.variable_scope('Encoder'):
        score_input = Input(batch_shape = (train_num,seq_num,features_out),name = 'score_input')     # Y_p
        mask_input = Input(batch_shape = (train_num,seq_num),name = 'mask_input')                    # 用于表征输入的时序，缺失的部分也会被标记为1
        missing_mask_input = Input(batch_shape = (train_num,seq_num),name = 'missing_mask_input')    # 在mask_inpur的基础上区分出了缺失，缺失的部分也会被标记为0，这个mask是用来在RNN中填充缺失的
        Fforward = []
        for i in range(mid_layer_num):
            Fforward.append(Dense(latent_dim, name = 'Feedforward_'+str(i),activation = act_fun))
        if view_num==2:
            Fforward.append(Dense(latent_dim+features_out+features_in[2], name = 'Feedforward_output'))           # 用于将隐状态映射成预测的评分
        else:
            Fforward.append(Dense(latent_dim+features_out, name = 'Feedforward_output'))           
        encoder_RNN = model_list[rnn_unit]

        if view_num==2:
            encoder_inputs = Lambda(lambda x: K.concatenate(x, axis=2),name = 'Concatenate_inputs_1')([encoder_inputs,Input_1])
        encoder_inputs = Lambda(lambda x: K.concatenate(x, axis=2),name = 'Concatenate_inputs_2')([encoder_inputs,score_input]) # s_t = [h_t,y_t]
        global latent_temp
        latent_temp = encoder_inputs
        all_encoder_outputs = []
        # 第一个时刻的输出
        # 取出对应的mask
        mask_temp = Lambda(tensor_slice,arguments={'ax':1,'ind':0})(mask_input)           # n x 1
        # 取出input切片
        input_temp = Lambda(tensor_slice,arguments={'ax':1,'ind':0})(encoder_inputs)      # n x 1 x (latent_dim + score_dim)
        if rnn_unit == 'LSTM':
            encoder_outputs, encoder_state_h, encoder_state_c = encoder_RNN(input_temp,mask = mask_temp)     # h_t = LSTM(x_t,0)
            states = [encoder_state_h,encoder_state_c] # 用于更新初始状态h_t
        else:
            encoder_outputs, encoder_state_h = encoder_RNN(input_temp,mask = mask_temp)     # h_t = LSTM(x_t,0)
            states = [encoder_state_h] # 用于更新初始状态h_t

        for layer_num in range(mid_layer_num):
            encoder_outputs = Fforward[layer_num](encoder_outputs)
        encoder_single_output = Add()([Fforward[-1](encoder_outputs),input_temp])      # x'_t = W * h_t + x_{t-1}

        all_encoder_outputs.append(encoder_single_output)

        for i in range(1,seq_num):
            # 把当前时刻的mask提取出来
            mask_temp = Lambda(tensor_slice,arguments={'ax':1,'ind':i})(mask_input)        # n x 1
            missing_mask_temp = Lambda(tensor_slice,arguments={'ax':1,'ind':i})(missing_mask_input)
            # 以上一时刻的输出以及当前时刻的输入作为总输入
            input_temp = Lambda(tensor_slice,arguments={'ax':1,'ind':i})(encoder_inputs)   # n x 1 x 129
            input_temp =HardCalculateLayer_3()([encoder_single_output,input_temp,missing_mask_temp])
            if rnn_unit == 'LSTM':
                encoder_outputs, encoder_state_h, encoder_state_c = encoder_RNN(input_temp,initial_state=states,mask = mask_temp)     # h_t = LSTM(x_t,0)
                states = [encoder_state_h,encoder_state_c] # 用于更新初始状态h_t
            else:
                encoder_outputs, encoder_state_h = encoder_RNN(input_temp,initial_state=states,mask = mask_temp)     # h_t = LSTM(x_t,0)
                states = [encoder_state_h] # 用于更新初始状态h_t

            # 输出状态经过Dense层预测下一时刻的输出
            for layer_num in range(mid_layer_num):
                encoder_outputs = Fforward[layer_num](encoder_outputs)
            encoder_single_output = Add()([Fforward[-1](encoder_outputs),input_temp])

            if i != seq_num-1:
                # 预测得到的输出需要放进loss优化
                all_encoder_outputs.append(encoder_single_output)

        encoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1),name = 'Concatenate_encoder')(all_encoder_outputs)
        # n x seq x 129
    #Decoder
    with tf.variable_scope('Decoder'):
        decoder_inputs = encoder_single_output
        all_outputs = []
        inputs = decoder_inputs                                                              # n x 1 x 1
        for _ in range(pred_length):
            # Run the decoder on one timestep
            if rnn_unit == 'LSTM':
                decoder_outputs, decoder_state_h, decoder_state_c = encoder_RNN(inputs,initial_state=states)     # h_t = LSTM(x_t,0)
                states = [decoder_state_h,decoder_state_c] # 用于更新初始状态h_t
            else:
                decoder_outputs, decoder_state_h = encoder_RNN(inputs,initial_state=states)     # h_t = LSTM(x_t,0)
                states = [decoder_state_h] # 用于更新初始状态h_t

            for layer_num in range(mid_layer_num):
                decoder_outputs = Fforward[layer_num](decoder_outputs)
            outputs = Add()([Fforward[-1](decoder_outputs),inputs])
            # Store the current prediction (we will concatenate all predictions later)
            all_outputs.append(outputs)
            inputs = outputs
        # Concatenate all predictions
        decoder_outputs_temp = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
        if view_num==2:
            decoder_outputs = [Lambda(tensor_slice,arguments={'ax':2,'ind':latent_dim+features_in[2]+i})(decoder_outputs_temp) for i in range(features_out)]
        else:
            decoder_outputs = [Lambda(tensor_slice,arguments={'ax':2,'ind':latent_dim+i})(decoder_outputs_temp) for i in range(features_out)]
    
    #总模型
    model_output = Recon_X+[encoder_outputs]+decoder_outputs
    model = Model([Input_1, score_input,mask_input,missing_mask_input],model_output)
    return model


# ## 2.4 预测与模型验证

# In[ ]:


def define_fusing_model(features_in,latent_dim,seq_num,train_num,view_num,act_fun):
    mid_layer_num = 1
    with tf.variable_scope('Input'):
        Input_1 = Input(batch_shape = (train_num,seq_num,features_in[2]))  # 实际上并不参与运算，只是用来表征维度的
    # CPM-Net
    with tf.variable_scope('FusionH'):
        encoder_inputs_H = LatentLayer(train_len = train_num,l_dim=latent_dim,name = 'LatentLayer_test')(Input_1)   # n x seq x 128
        global latent_temp_H
        latent_temp_H = encoder_inputs_H

    with tf.variable_scope('FusionDegration'):
        De_layer = []      # Degradation layers
        for i in range(view_num):
            temp_de_layer = []
            for j in range(mid_layer_num):
                temp_de_layer.append(Dense(latent_dim, name = 'Degeneration_Layer_view'+str(i+1)+'_'+str(j), activation = act_fun,trainable = False))
            temp_de_layer.append(Dense(features_in[i], name = 'Degeneration_Layer_view'+str(i+1)+'_output', activation = act_fun,trainable = False))
            De_layer.append(temp_de_layer)
        Recon_X = []

        for i in range(view_num):
            temp_input = De_layer[i][0](encoder_inputs_H)
            for j in range(1,mid_layer_num):
                temp_input = De_layer[i][j](temp_input)
            Recon_X.append(De_layer[i][-1](temp_input))  # shape: n x seq x dv

    fusion_model_output = Recon_X+[encoder_inputs_H]
    fusion_model = Model([Input_1],fusion_model_output)
    return fusion_model

def define_testing_model(features_in,features_out,latent_dim,seq_num,pred_length,view_num,train_num,rnn_unit,act_fun):
    mid_layer_num = 1
    model_list ={'LSTM':LSTM(latent_dim,return_state = True,return_sequences = True, name = 'encoder_LSTM',trainable=False),
                 'GRU':GRU(latent_dim,return_state = True,return_sequences = True, name = 'encoder_GRU',trainable=False)}
    
    with tf.variable_scope('Input'):
        Input_1 = Input(batch_shape = (train_num,seq_num,features_in[2]))  # Demographics
    
    with tf.variable_scope('Encoder'):
        encoder_input_H = Input(batch_shape = (train_num,seq_num,latent_dim),name = 'encoder_input')
        score_input = Input(batch_shape = (train_num,seq_num,features_out),name = 'score_input')
        mask_input = Input(batch_shape = (train_num,seq_num),name = 'mask_input',dtype = bool)
        missing_mask_input = Input(batch_shape = (train_num,seq_num),name = 'missing_mask_input')
        # Encoder
        encoder_RNN = model_list[rnn_unit]
        Fforward = []
        for i in range(mid_layer_num):
            Fforward.append(Dense(latent_dim, name = 'Feedforward_'+str(i),activation = act_fun,trainable=False))
        if view_num==2:
            Fforward.append(Dense(latent_dim+features_out+features_in[2], name = 'Feedforward_output',trainable=False))
        else:
            Fforward.append(Dense(latent_dim+features_out, name = 'Feedforward_output',trainable=False))

        temp_input_H = encoder_input_H
        if view_num==2:
            temp_input_H= Lambda(lambda x: K.concatenate(x, axis=2),name = 'Concatenate_input_1')([encoder_input_H,Input_1])
        encoder_input = Lambda(lambda x: K.concatenate(x, axis=2),name = 'Concatenate_input_2')([temp_input_H,score_input])


        all_encoder_outputs = []
        # 第一个时刻的输出

        # 取出对应的mask
        mask_temp = Lambda(tensor_slice,arguments={'ax':1,'ind':0})(mask_input)           # n x 1
        # 取出input切片
        input_temp = Lambda(tensor_slice,arguments={'ax':1,'ind':0})(encoder_input)      # n x 1 x 128
        if rnn_unit == 'LSTM':
            encoder_outputs, encoder_state_h, encoder_state_c = encoder_RNN(input_temp,mask = mask_temp)     # h_t = LSTM(x_t,0)
            states = [encoder_state_h,encoder_state_c] # 更新初始状态h_t
        else:
            encoder_outputs, encoder_state_h = encoder_RNN(input_temp,mask = mask_temp)     # h_t = LSTM(x_t,0)
            states = [encoder_state_h] # 更新初始状态h_t

        for layer_num in range(mid_layer_num):
            encoder_outputs = Fforward[layer_num](encoder_outputs)
        encoder_single_output = Add()([Fforward[-1](encoder_outputs),input_temp])      # x'_t = W * h_t + x_{t-1}
        
        all_encoder_outputs.append(encoder_single_output)
        for i in range(1,seq_num):
            # 把当前时刻的mask提取出来
            mask_temp = Lambda(tensor_slice,arguments={'ax':1,'ind':i})(mask_input)        # n x 1
            missing_mask_temp = Lambda(tensor_slice,arguments={'ax':1,'ind':i})(missing_mask_input)
            # 以上一时刻的输出以及当前时刻的输入作为总输入
            input_temp = Lambda(tensor_slice,arguments={'ax':1,'ind':i})(encoder_input)   # n x 1 x 128
            input_temp =HardCalculateLayer_3()([encoder_single_output,input_temp,missing_mask_temp])
            if rnn_unit == 'LSTM':
                encoder_outputs, encoder_state_h, encoder_state_c = encoder_RNN(input_temp,initial_state=states,mask = mask_temp)     # h_t = LSTM(x_t,0)
                states = [encoder_state_h,encoder_state_c] # 更新初始状态h_t
            else:
                encoder_outputs, encoder_state_h = encoder_RNN(input_temp,initial_state=states,mask = mask_temp)     # h_t = LSTM(x_t,0)
                states = [encoder_state_h] # 更新初始状态h_t


            # 输出状态经过Dense层预测下一时刻的输出
            for layer_num in range(mid_layer_num):
                encoder_outputs = Fforward[layer_num](encoder_outputs)
            encoder_single_output = Add()([Fforward[-1](encoder_outputs),input_temp])

            if i != seq_num-1:
                # 预测得到的输出需要放进loss优化
                all_encoder_outputs.append(encoder_single_output)

    #Decoder
    with tf.variable_scope('Decoder'):
        decoder_inputs = encoder_single_output

        all_outputs = []
        inputs = decoder_inputs                                                              # n x 1 x 1
        for _ in range(pred_length):
            # Run the decoder on one timestep
            if rnn_unit == 'LSTM':
                decoder_outputs, decoder_state_h, decoder_state_c = encoder_RNN(inputs,initial_state=states)     # h_t = LSTM(x_t,0)
                states = [decoder_state_h, decoder_state_c] # 更新初始状态h_t
            else:
                decoder_outputs, decoder_state_h = encoder_RNN(inputs,initial_state=states)     # h_t = LSTM(x_t,0)
                states = [decoder_state_h] # 更新初始状态h_t

            for layer_num in range(mid_layer_num):
                decoder_outputs = Fforward[layer_num](decoder_outputs)
            outputs = Add()([Fforward[-1](decoder_outputs),inputs])
            # Store the current prediction (we will concatenate all predictions later)
            all_outputs.append(outputs)
            inputs = outputs

        # Concatenate all predictions
        decoder_outputs_temp = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
        if view_num==2:
            decoder_outputs = [Lambda(tensor_slice,arguments={'ax':2,'ind':latent_dim+features_in[2]+i})(decoder_outputs_temp) for i in range(features_out)]
        else:
            decoder_outputs = [Lambda(tensor_slice,arguments={'ax':2,'ind':latent_dim+i})(decoder_outputs_temp) for i in range(features_out)]

    #总模型
    pred_model = Model([Input_1,encoder_input_H, score_input, mask_input,missing_mask_input],decoder_outputs)
    return pred_model


# # （3）模型定义完毕后，导入数据开始训练
# In[ ]:
def model_tuner(args):
    global X_data_train,X_mask_train,X_missing_mask_train,X_score_train,Y_train,train_num,latent_temp,latent_dim,epochs,rnn_unit,view_num

    latent_dim= args['ldim']
    learning_rate = args['lr']
    decay = args['decay']
    weight_fus = args['w_fus']
    weight_rec = args['w_rec']
    X_data_val_temp = [X_data_train[i][round(train_num*0.8):,:] for i in range(len(X_data_train))]
    X_data_train_temp = [X_data_train[i][0:round(train_num*0.8),:] for i in range(len(X_data_train))]

    X_mask_train_temp = X_mask_train[0:round(train_num*0.8),:]
    X_mask_val_temp = X_mask_train[round(train_num*0.8):,:]

    X_missing_mask_train_temp = X_missing_mask_train[0:round(train_num*0.8),:]
    X_missing_mask_val_temp = X_missing_mask_train[round(train_num*0.8):,:]

    X_score_train_temp = X_score_train[0:round(train_num*0.8),:,:]
    X_score_val_temp = X_score_train[round(train_num*0.8):,:,:]

    # X_data_train_temp.append(X_score_train_temp)
    # X_data_val_temp.append(X_score_val_temp)

    Y_train_temp = Y_train[0:round(train_num*0.8),:,:]
    Y_val_temp = Y_train[round(train_num*0.8):,:,:]

    features_in = [X_data_train_temp[i].shape[2] for i in range(len(X_data_train_temp))]
    features_out = Y_train_temp.shape[2]

    seq_num = X_data_train_temp[0].shape[1]
    pred_length = Y_train_temp.shape[1]
    new_train_num = X_data_train_temp[0].shape[0]
    new_test_num = X_data_val_temp[0].shape[0]

    train_model = define_training_model(features_in,features_out,latent_dim,seq_num,pred_length,new_train_num,rnn_unit,view_num,'relu')
    adam_opt = keras.optimizers.Adam(lr=learning_rate,decay=decay)
    loss_list_train = []
    weight_list_train = []
    for i in range(view_num):
        loss_list_train.append(met.new_MSE)       # loss for reconstraction
        weight_list_train.append(weight_fus)
    loss_list_train.append(met.MAE)                # loss for fitting
    weight_list_train.append(weight_rec)
    for i in range(Y_train_temp.shape[2]):
        loss_list_train.append(met.MAE)            # loss for prediction
        weight_list_train.append(1)
    train_model.compile(optimizer = adam_opt,loss = loss_list_train,loss_weights = weight_list_train)

    lh = cbk.LearningHandler(
        lr=learning_rate,
        drop=0.1,
        lr_tensor=train_model.optimizer.lr,
        patience=10)
    # MMSE/CDR_Global/CDR_SOB/ADAS-Cog
    true_output = X_data_train_temp[:view_num]+[latent_temp[:,1:,:]]+[Y_train_temp[:,:,i][:,:,np.newaxis] for i in range(Y_train_temp.shape[2])]
    history = train_model.fit([X_data_train_temp[2],X_score_train_temp,X_mask_train_temp,X_missing_mask_train_temp], true_output, epochs=epochs,steps_per_epoch = 1, verbose = 0,callbacks=[lh])
    # print('=========> Iter',turn,': Train model traing complete.')

    # 保存训练得到的权重
    train_model.save_weights('./saved_model/tuner_model'+'_fixed2')

    # 构建测试模型，并导入训练得到的权重
    fusion_model = define_fusing_model(features_in,latent_dim,seq_num,new_test_num,view_num,'relu')
    adam_opt = keras.optimizers.Adam(lr=learning_rate,decay=decay)

    loss_list_fuse = []
    weight_list_fuse = []
    for i in range(view_num):
        loss_list_fuse.append(met.new_MSE)       # loss for reconstraction
        weight_list_fuse.append(weight_fus)
    loss_list_fuse.append(met.new_MSE)                # loss for fitting
    weight_list_fuse.append(1)
    fusion_model.compile(optimizer = adam_opt,loss = loss_list_fuse,loss_weights = weight_list_fuse)
    # print('=========> Iter',turn,': Fusion model compiling succeed.')
    test_model = define_testing_model(features_in,features_out,latent_dim,seq_num,pred_length,view_num,new_test_num,rnn_unit,'relu')
    fusion_model.load_weights('./saved_model/tuner_model'+'_fixed2',by_name=True)
    test_model.load_weights('./saved_model/tuner_model'+'_fixed2',by_name=True)


    # 训练测试模型，学H
    lh = cbk.LearningHandler(
        lr=learning_rate,
        drop=0.1,
        lr_tensor=fusion_model.optimizer.lr,
        patience=10)
    # print('=========> Iter',turn,': Start fusion model traing...')
    fus_output = X_data_val_temp[:view_num] + [latent_temp_H]
    history = fusion_model.fit([X_data_val_temp[2]], fus_output, epochs=100,steps_per_epoch = 1, verbose = 0,callbacks=[lh])
    # print('=========> Iter',turn,': Fusion model traing complete.')
    fus_pred_output = fusion_model.predict([X_data_val_temp[2]],batch_size = new_test_num)
    latent_rep = fus_pred_output[-1]
    # 模型测试
    res =test_model.predict([X_data_val_temp[2],latent_rep,X_score_val_temp,X_mask_val_temp,X_missing_mask_val_temp],batch_size = new_test_num)
    res_score = np.stack(res)
    target_score = Y_val_temp[:,:,:,np.newaxis]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        predx_score = tf.cast(res_score,dtype = tf.float32)
        targetx_score = tf.cast(target_score,dtype = tf.float32)
        MAE_value_MMSE = met.MAE(targetx_score[:,:,0,:],predx_score[0,:,:,:]).eval()

    tf.reset_default_graph()
    K.clear_session()
    return MAE_value_MMSE


# In[ ]:


def model_evaluation(f_name,saved_filename_temp,turn,train_idxes,test_idxes):
    global X_data_train,X_mask_train,X_missing_mask_train,X_score_train,Y_train,train_num,latent_dim,epochs,rnn_unit,data_view_num,maxeval,seq_num,view_num
    data = sio.loadmat(f_name)
    train_index_temp = train_idxes[turn]
    test_index_temp = test_idxes[turn]
    train_index = [i-1 for i in train_index_temp]
    test_index = [i-1 for i in test_index_temp]

    # 多模态特征
    X_data = [np.stack(data['X_data'][:,i]) for i in range(data['X_data'].shape[1])]
    X_data_train = [X_data[i][:,train_index] for i in range(len(X_data))]
    X_data_test = [X_data[i][:,test_index] for i in range(len(X_data))]
    # 掩模
    X_mask_train = data['X_mask'][train_index,:]
    X_mask_test = data['X_mask'][test_index,:]

    X_missing_mask_train = data['X_missing_mask'][train_index,:]
    X_missing_mask_test = data['X_missing_mask'][test_index,:]
    # 特征时序的评分
    X_score = [np.stack(data['X_score'][:,i]) for i in range(data['X_score'].shape[1])][0]
    X_score_train = X_score[:,train_index,:]
    X_score_test = X_score[:,test_index,:]

    if data_view_num == 4:
        X_data_train.append(X_score_train)
        X_data_test.append(X_score_test)
    else:
        X_data_train = X_data_train[:data_view_num]
        X_data_test = X_data_test[:data_view_num]

    # 标签
    Y = [np.stack(data['gt'][:,i]) for i in range(data['gt'].shape[1])][0]
    Y_train = Y[:,train_index,:]
    Y_test = Y[:,test_index,:]

    # 尺寸转换为 n x seq x dim
    X_data_train = [X_data_train[i].swapaxes(0,1) for i in range(len(X_data_train))]
    X_data_test = [X_data_test[i].swapaxes(0,1) for i in range(len(X_data_test))]
    X_score_train = X_score_train.swapaxes(0,1)
    X_score_test = X_score_test.swapaxes(0,1)
    Y_train = Y_train.swapaxes(0,1)
    Y_test = Y_test.swapaxes(0,1)

    features_in = [X_data_train[i].shape[2] for i in range(len(X_data_train))]
    features_out = Y_train.shape[2]

    seq_num = X_data_train[0].shape[1]
    pred_length = Y_train.shape[1]
    train_num = X_data_train[0].shape[0]
    test_num = X_data_test[0].shape[0]

    #%% 填补缺失值
    for i in range(data_view_num):
        X_data_train[i][np.isnan(X_data_train[i])] = -5
        X_data_test[i][np.isnan(X_data_test[i])] = -5
    Y_train[np.isnan(Y_train)] = -5
    Y_test[np.isnan(Y_test)] = -5
    X_score_train[np.isnan(X_score_train)] = -5
    X_score_test[np.isnan(X_score_test)] = -5

    space = {
        'ldim': hp.choice('ldim',[64,128,256]),
        'lr':hp.uniform('lr',0.001,0.1),
        'decay': hp.uniform('decay',0.001,0.1),
        'w_fus':hp.uniform('w_fus',0.01,1),
        'w_rec':hp.uniform('w_rec',0.1,10)
    }



    best = fmin(fn=model_tuner, space=space,algo=rand.suggest, max_evals=maxeval)


    #%% 选择好参数后，带入模型正式训练
    act_fun = 'relu'
    para_ldim = [64,128,256]
    latent_dim = para_ldim[best['ldim']]
    learning_rate = best['lr']
    decay = best['decay']
    weight_fus = best['w_fus']
    weight_rec = best['w_rec']
    best_para = [latent_dim,learning_rate,decay,weight_fus,weight_rec]


    #%%% 构建模型
    print('=========> Iter',turn,': Start constructing and compiling train model...')
    train_model = define_training_model(features_in,features_out,latent_dim,seq_num,pred_length,train_num,rnn_unit,view_num,act_fun)
    adam_opt = keras.optimizers.Adam(lr=learning_rate,decay=decay)
    loss_list_train = []
    weight_list_train = []
    for i in range(view_num):
        loss_list_train.append(met.new_MSE)       # loss for reconstraction
        weight_list_train.append(weight_fus)
    loss_list_train.append(met.MAE)                # loss for fitting
    weight_list_train.append(weight_rec)
    for i in range(Y_train.shape[2]):
        loss_list_train.append(met.MAE)            # loss for prediction
        weight_list_train.append(1)
    train_model.compile(optimizer = adam_opt,loss = loss_list_train,loss_weights = weight_list_train)
    ## 训练模型
    # 回调函数，用于调整学习率和早停
    lh = cbk.LearningHandler(
        lr=learning_rate,
        drop=0.1,
        lr_tensor=train_model.optimizer.lr,
        patience=10)
    # 回调函数，用于保存loss和梯度直方图
    tbCallBack = TensorBoard(log_dir='./TensorBoard',  # log 目录
                             histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             #                  batch_size=32,     # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True, # 是否可视化梯度直方图
                             write_images=True)
    print('=========> Iter',turn,': Start train model traing...')
    true_output = X_data_train[:view_num]+[latent_temp[:,1:,:]]+[Y_train[:,:,i][:,:,np.newaxis] for i in range(Y_train.shape[2])]
    train_model.fit([X_data_train[2],X_score_train,X_mask_train,X_missing_mask_train], true_output, epochs=epochs,steps_per_epoch = 1, verbose = 0,callbacks=[lh])
    print('=========> Iter',turn,': Train model traing complete.')

    # 保存训练得到的权重
    train_model.save_weights('./saved_model/'+saved_filename_temp+'Turn_'+str(turn)+'_fixed2')

    # 构建测试模型，并导入训练得到的权重
    print('=========> Iter',turn,': Start constructing and compiling fusion model...')
    fusion_model = define_fusing_model(features_in,latent_dim,seq_num,test_num,view_num,act_fun)
    adam_opt = keras.optimizers.Adam(lr=learning_rate,decay=decay)

    loss_list_fuse = []
    weight_list_fuse = []
    for i in range(view_num):
        loss_list_fuse.append(met.new_MSE)       # loss for reconstraction
        weight_list_fuse.append(weight_fus)
    loss_list_fuse.append(met.new_MSE)                # loss for fitting
    weight_list_fuse.append(1)
    fusion_model.compile(optimizer = adam_opt,loss = loss_list_fuse,loss_weights = weight_list_fuse)
    # print('=========> Iter',turn,': Fusion model compiling succeed.')
    test_model = define_testing_model(features_in,features_out,latent_dim,seq_num,pred_length,view_num,test_num,rnn_unit,act_fun)
    fusion_model.load_weights('./saved_model/'+saved_filename_temp+'Turn_'+str(turn)+'_fixed2',by_name=True)
    test_model.load_weights('./saved_model/'+saved_filename_temp+'Turn_'+str(turn)+'_fixed2',by_name=True)

    # 训练测试模型，学H
    lh = cbk.LearningHandler(
        lr=learning_rate,
        drop=0.1,
        lr_tensor=fusion_model.optimizer.lr,
        patience=10)
    print('=========> Iter',turn,': Start fusion model traing...')
    fus_output = X_data_test[:view_num] + [latent_temp_H]
    fusion_model.fit([X_data_test[2]], fus_output, epochs=100,steps_per_epoch = 1, verbose = 0,callbacks=[lh])
    print('=========> Iter',turn,': Fusion model traing complete.')
    fus_pred_output = fusion_model.predict([X_data_test[2]],batch_size = test_num)
    latent_rep = fus_pred_output[-1]
    # 模型测试
    res =test_model.predict([X_data_test[2],latent_rep,X_score_test,X_mask_test,X_missing_mask_test],batch_size = test_num)
    res_score = np.stack(res)
    target_score = Y_test[:,:,:,np.newaxis]
    
    return res_score,target_score

# In[ ]:
if __name__ == "__main__":
    global rnn_unit,data_view_num,epochs,maxeval,view_num
    data_view_num = 3       # overall modalities
    view_num = 3            # for fusion
    epochs = 300            # 400
    maxeval = 20            # hyperopt max eval times
    kfold_num = 10          # cv fold number
    iter_time = 10           # random run times
    for rnn_unit in ['LSTM']:
        f_name = './Data/demo_data.mat' 
        saved_filename_temp = 'Fusion_' + rnn_unit +'_MF3_View'+str(data_view_num)+'_'
        if not os.path.exists('./saved_model/'):
            os.makedirs('./saved_model/')

        for iters in range(iter_time):
            train_idxes = []
            test_idxes = []
            data = sio.loadmat(f_name)
            kf = KFold(n_splits = kfold_num,shuffle = True)
            for train_idx,test_idx in kf.split(data['X_mask']):
                train_idxes.append(train_idx)
                test_idxes.append(test_idx)
            y_pred = []
            y_true = []
            total_starttime = datetime.datetime.now()
            for i in range(kfold_num):
                starttime = datetime.datetime.now()
                res_score,target_score = model_evaluation(f_name,saved_filename_temp,i,train_idxes,test_idxes)
                y_pred.append(res_score)
                y_true.append(target_score)
                endtime = datetime.datetime.now()
                elapsed_sec = (endtime - starttime).total_seconds()
                y_pred[i] = y_pred[i].transpose(1,2,0,3)
                saved_filename = saved_filename_temp+str(iters)+'.mat'
                sio.savemat(os.getcwd()+'/'+saved_filename, {
                                    'y_true':y_true,
                                    'y_pred':y_pred})

            total_endtime = datetime.datetime.now()
            total_elapsed_sec = (total_endtime - total_starttime).total_seconds()
