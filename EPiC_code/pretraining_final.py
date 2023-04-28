import numpy as np
import math
import random
from math import log, e
from tqdm import tqdm
from scipy.stats import *
import copy
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os
import argparse
import utils
import ecg_net
import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import parsing
import shutil

parser = parsing.create_parser()
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}



def train_step(inputs, labels, model, optimizer):

    optimizer.zero_grad()

    outputs  = model(inputs)

    loss = nn.MSELoss()(outputs, labels)

    loss.backward()

    optimizer.step()


    return loss.item()


def test_step(inputs, model):

    outputs  = model(inputs)
    outputs = outputs.detach().cpu().clone().numpy()


    return outputs


def pre_train(Net, train_dataset):


    train_loss_epoch   = np.ones((args.epochs,  1))

    '''
    ***Start of Training***
    '''

    for epoch in range(args.epochs):

        train_loss_batch = []

        lr = args.lr
        optimizer = optim.Adam(Net.parameters(), lr=lr)

        Net.train()


        for data_batch, label_batch in train_dataset:

            data_batch, label_batch =  map(lambda x: x.to(device), (data_batch, label_batch))

            loss = train_step(data_batch, label_batch, Net, optimizer)

            train_loss_batch.append(loss)


        train_loss_epoch[epoch] = utils.Average(train_loss_batch)

        
        print('epoch: {}'.format(epoch), 'train_mse_loss: {}'.format(train_loss_epoch[epoch]))

    return Net, train_loss_epoch




if __name__ == '__main__':


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #  
    signal_name = args.modality
    emotion_name = args.emotion

    scenario_num = args.scenario
    fold_num = args.fold

    if scenario_num ==1: 
        result_path = utils.parent_path + '/EPiC_result/final' + '/scenario_{}/'.format(scenario_num)
    else: 
        result_path = utils.parent_path + '/EPiC_result/final' + '/scenario_{}/fold_{}/'.format(scenario_num, fold_num)

    
    utils.make_dir(result_path)


    if scenario_num == 1: 
        train_path, test_path, _, _, _, _ = train_test_split.load_data(scenario_num)
    else: 
        train_path, test_path, _, _, _, _ = train_test_split.load_data(scenario_num, fold_num)


    '''
    Indepdendent training , do not change this part ------START
    '''



    train_entire_signal = np.load(train_path + '/{}/'.format(signal_name)   + 'sliding_window/' + 'entire.npy')
    train_entire_label  = np.load(train_path + '/labels/'                   + 'sliding_window/' + 'entire.npy')


    # print(train_entire_ecg.shape, test_entire_ecg.shape, train_entire_label.shape, test_entire_label.shape)

    train_entire_signal = stats.zscore(train_entire_signal, axis=1)


    # emotion_label = ['valence', 'arousal']
    if emotion_name == 'valence':
        train_label = train_entire_label[:,0]

    else: # arousal
        train_label = train_entire_label[:,1] 


    scaled_train_label = utils.min_max_scale(train_label)


    # scaled_back_train_label = utils.min_max_inverse_scale(train_label, scaled_train_label)


    Net = ecg_net.Conv_EEG().to(device)   
    Net.apply(utils.WeightInit)
    Net.apply(utils.WeightClipper)

    print(Net)
    
    train_entire_signal = np.expand_dims(train_entire_signal, axis=1)
    scaled_train_label  = np.expand_dims(scaled_train_label, axis=1)

        
    train_dataset = utils.load_dataset_to_device(train_entire_signal, scaled_train_label,  batch_size=64,  shuffle_flag=True)
    
    Net, train_loss = pre_train(Net, train_dataset)


    '''save training loss and model checkpoint of the last epoch'''
    loss_path = train_path + '/{}/'.format(signal_name) + 'sliding_window/' + '{}_'.format(emotion_name) + 'train_loss.csv'
    np.savetxt(loss_path, train_loss, delimiter=',') 

    ckpt_path = train_path + '/{}/'.format(signal_name) + 'sliding_window/' + '{}_'.format(emotion_name) + 'model' 
    torch.save(Net.state_dict(), ckpt_path)






