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
import pandas as pd
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



def post_test(Net, test_dataset):

    test_pred_batch = []


    Net.eval()

    with torch.no_grad():

        for data_batch, _ in test_dataset:

            data_batch =  data_batch.to(device)

            pred = test_step(data_batch, Net)

            test_pred_batch.append(pred)


    test_pred_batch = [item for sublist in test_pred_batch for item in sublist]
    test_pred_batch  = utils.min_max_inverse_scale(test_pred_batch)

    return test_pred_batch

def train(Net, train_dataset, test_dataset):


    train_loss_epoch   = np.ones((args.epochs,  1))

    '''
    ***Start of Training***
    '''

    for epoch in range(args.epochs):
        start = time.time()
        train_loss_batch = []
        test_pred_batch  = []


        lr = args.lr

        # optimizer = optim.SGD(Net.parameters(), lr=lr)
        optimizer = optim.Adam(Net.parameters(), lr=lr)


        Net.train()


        for data_batch, label_batch in train_dataset:

            data_batch, label_batch =  map(lambda x: x.to(device), (data_batch, label_batch))

            loss = train_step(data_batch, label_batch, Net, optimizer)

            train_loss_batch.append(loss)


        train_loss_epoch[epoch] = utils.Average(train_loss_batch)


        Net.eval()

        with torch.no_grad():

            for data_batch, _ in test_dataset:

                data_batch =  data_batch.to(device)

                pred = test_step(data_batch, Net)

                test_pred_batch.append(pred)


        test_pred_batch = [item for sublist in test_pred_batch for item in sublist]


        # plt.plot(np.asarray(test_pred_batch), label='pred')
    
        # plt.legend()
        # plt.show()


        
        print('epoch: {}'.format(epoch), 'train_mse_loss: {}'.format(train_loss_epoch[epoch]))

        
    test_pred_batch  = utils.min_max_inverse_scale(test_pred_batch)
    

    return test_pred_batch


if __name__ == '__main__':


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #  
    signal_name = args.modality
    emotion_name = args.emotion

    scenario_num = args.scenario
    fold_num = args.fold


    if args.use_pretrain == True: 
        method = '/retrain'
    else:
        method = '/scratch'

    if scenario_num ==1: 
        result_path =  utils.parent_path + '/EPiC_result/final'  + '/{}'.format(signal_name) +  method + '/scenario_{}/'.format(scenario_num)
    else: 
        result_path =  utils.parent_path + '/EPiC_result/final'  + '/{}'.format(signal_name) +  method + '/scenario_{}/fold_{}/'.format(scenario_num, fold_num)


    utils.make_dir(result_path)



    utils.set_permissions_recursive(result_path)

    if scenario_num == 1: 
        train_path, test_path, _, _, _, _ = train_test_split.load_data(scenario_num)
    else: 
        train_path, test_path, _, _, _, _ = train_test_split.load_data(scenario_num, fold_num)



    '''
    Subject and session depdendent training , do not change this part ------START
    '''

    train_signal_path   = train_path + '/{}/'.format(signal_name) + 'sliding_window/'
    test_signal_path    = test_path + '/{}/'.format(signal_name) + 'sliding_window/'

    train_label_path    = train_path + '/labels/' + 'sliding_window/'
    test_label_path     = test_path + '/labels/' + 'sliding_window/'


    final_label_path    = test_path + '/annotations/'    # Do not write this folder


    files = os.listdir(train_label_path)
    files.remove('entire.npy')



    if scenario_num ==1:

        for file_name in tqdm(files):


            Net = ecg_net.Conv_EEG().to(device)
            if args.use_pretrain == True: 
                ckpt_path = train_path + '/{}/'.format(signal_name) + 'sliding_window/' + '{}_'.format(emotion_name) + 'model' 
                Net.load_state_dict(torch.load(ckpt_path))
            else: 
                Net.apply(utils.WeightInit)
                Net.apply(utils.WeightClipper)
                

            train_individual_signal = np.load(train_signal_path  + file_name)
            test_individual_signal  = np.load(test_signal_path   + file_name)

            train_individual_label  = np.load(train_label_path   + file_name)
            test_individual_label   = np.load(test_label_path    + file_name)


            train_individual_signal = stats.zscore(train_individual_signal, axis=1)
            test_individual_signal  = stats.zscore(test_individual_signal, axis=1)

            # plt.plot(train_individual_signal[2])
            # plt.plot(test_individual_signal[1])

            # plt.show()

            # emotion_label = ['valence', 'arousal']
            if emotion_name == 'valence':
                train_label = train_individual_label[:,0]
                test_label  = test_individual_label[:,0]

            else: # arousal
                train_label = train_individual_label[:,1] 
                test_label  = test_individual_label[:,1] 


            scaled_train_label = utils.min_max_scale(train_label)
            scaled_test_label  = utils.min_max_scale(test_label)

            # print(len(train_individual_signal), len(scaled_train_label))
            train_individual_signal, test_individual_signal = np.expand_dims(train_individual_signal, axis=1),  np.expand_dims(test_individual_signal, axis=1)
            scaled_train_label,     scaled_test_label       = np.expand_dims(scaled_train_label,      axis=1),  np.expand_dims(scaled_test_label,      axis=1)

            # print(file_name, len(scaled_test_label), scaled_test_label[3:5]) 

            train_dataset = utils.load_dataset_to_device(train_individual_signal, scaled_train_label,  batch_size=16, shuffle_flag=True)
            test_dataset  = utils.load_dataset_to_device(test_individual_signal,   scaled_test_label,  batch_size=16, shuffle_flag=False)


            pred = train(Net, train_dataset, test_dataset)
            pred = pred.flatten().tolist()
            

            #Write result
            if os.path.exists(result_path + file_name.replace('.npy', '.csv')):
                try:
                    df = pd.read_csv(result_path + file_name.replace('.npy', '.csv'))
                    pass    
                except pd.errors.EmptyDataError:
                    shutil.copy(final_label_path + file_name.replace('.npy', '.csv'), result_path + file_name.replace('.npy', '.csv'))
                            
            else:
                shutil.copy(final_label_path + file_name.replace('.npy', '.csv'), result_path + file_name.replace('.npy', '.csv'))


            utils.write_result(emotion_name, pred, result_path + file_name.replace('.npy', '.csv'))

            # utils.set_permissions_recursive(result_path)



    elif scenario_num ==2: 
        # Session dependent and has to be Subject-independent
        session_list = [0, 2, 9, 10, 11, 13, 14, 20]

        train_files = os.listdir(train_label_path)
        if 'entire.npy' in train_files:
            train_files.remove('entire.npy')
        # train_sub_list =utils.unique([file_name.split('_')[1] for file_name in train_files])    

        test_files = os.listdir(test_label_path)
        if 'entire.npy' in test_files: 
            test_files.remove('entire.npy')
        # test_sub_list =utils.unique([file_name.split('_')[1] for file_name in test_files])


        for session_num in session_list:
            # print(train_files)
            
            train_individual_signal = np.zeros((0, 2000)) # 10s window * downsampled 200Hz data
            train_individual_label = np.zeros((0, 2))
            test_individual_signal = np.zeros((0, 2000)) # 10s window * downsampled 200Hz data
            test_individual_label = np.zeros((0, 2))
            

            train_file_list  = [x for x in train_files if 'vid_{}.'.format(session_num) in x]
            test_file_list   = [x for x in test_files  if 'vid_{}.'.format(session_num) in x]
        
            for file_name in train_file_list:
                train_individual_signal = np.vstack((train_individual_signal, np.load(train_signal_path  + file_name)))
                train_individual_label  = np.vstack((train_individual_label,  np.load(train_label_path   + file_name)))

 
            Net = ecg_net.Conv_EEG().to(device)

            if args.use_pretrain == True: 
                ckpt_path = train_path + '/{}/'.format(signal_name) + 'sliding_window/' + '{}_'.format(emotion_name) + 'model' 
                Net.load_state_dict(torch.load(ckpt_path))
            else: 
                Net.apply(utils.WeightInit)
                Net.apply(utils.WeightClipper)

            train_individual_signal = stats.zscore(train_individual_signal, axis=1)
            train_individual_signal = np.expand_dims(train_individual_signal, axis=1)  
    

            if emotion_name == 'valence':
                train_label = train_individual_label[:,0]
            else: # arousal
                train_label = train_individual_label[:,1] 


            scaled_train_label = utils.min_max_scale(train_label)
            scaled_train_label = np.expand_dims(scaled_train_label,   axis=1)

            train_dataset = utils.load_dataset_to_device(train_individual_signal, scaled_train_label,  batch_size=64, shuffle_flag=True)

            Net, _ = pre_train(Net, train_dataset)



            '''Use the pre-trained Net for each session of the same Subject'''

            '''Test Start'''
            for file_name in test_file_list:    
                test_individual_signal   = np.load(test_signal_path    + file_name)
                test_individual_label    = np.load(test_label_path     + file_name)

                test_individual_signal   = stats.zscore(test_individual_signal, axis=1)
                
                

                if emotion_name == 'valence':
                    test_label   = test_individual_label[:,0]
                else: # arousal
                    test_label   = test_individual_label[:,1] 

                test_individual_signal = np.expand_dims(test_individual_signal, axis=1)
    
                scaled_test_label   = utils.min_max_scale(test_label)
                scaled_test_label   =   np.expand_dims(scaled_test_label,   axis=1)
            
                test_dataset  = utils.load_dataset_to_device(test_individual_signal,   scaled_test_label,    batch_size=64, shuffle_flag=False)


                pred = post_test(Net, test_dataset)

                pred = pred.flatten().tolist()

                # utils.set_permissions_recursive(result_path, 0o2700)

                 #Write result
                if os.path.exists(result_path + file_name.replace('.npy', '.csv')):
                    try:
                        df = pd.read_csv(result_path + file_name.replace('.npy', '.csv'))
                        pass    
                    except pd.errors.EmptyDataError:
                        shutil.copy(final_label_path + file_name.replace('.npy', '.csv'), result_path + file_name.replace('.npy', '.csv'))
                                
                else:
                    shutil.copy(final_label_path + file_name.replace('.npy', '.csv'), result_path + file_name.replace('.npy', '.csv'))

                utils.write_result(emotion_name, pred, result_path + file_name.replace('.npy', '.csv'))



    elif scenario_num in [3,4]: 
        # Subject dependent 
    
        train_files = os.listdir(train_label_path)
        if 'entire.npy' in train_files:
            train_files.remove('entire.npy')
        test_files   = os.listdir(test_label_path)
        if 'entire.npy' in test_files:
            test_files.remove('entire.npy')

        sub_list =utils.unique([file_name.replace('.csv', '').split('_')[1] for file_name in train_files])

        # print(len(sub_list))

        for sub in sub_list:
            # print(train_files)
            
            train_individual_signal = np.zeros((0, 2000)) # 10s window * downsampled 200Hz data
            train_individual_label = np.zeros((0, 2))
            test_individual_signal = np.zeros((0, 2000)) # 10s window * downsampled 200Hz data
            test_individual_label = np.zeros((0, 2))


            train_file_list  = [x for x in train_files if 'sub_' + sub + '_' in x]
            test_file_list   = [x for x in test_files  if 'sub_' + sub + '_'in x]


            '''Training Start'''
        
            for file_name in train_file_list:
                train_individual_signal = np.vstack((train_individual_signal, np.load(train_signal_path  + file_name)))
                train_individual_label  = np.vstack((train_individual_label,  np.load(train_label_path   + file_name)))


            Net = ecg_net.Conv_EEG().to(device)
            if args.use_pretrain == True: 
                ckpt_path = train_path + '/{}/'.format(signal_name) + 'sliding_window/' + '{}_'.format(emotion_name) + 'model' 
                Net.load_state_dict(torch.load(ckpt_path))
            else: 
                Net.apply(utils.WeightInit)
                Net.apply(utils.WeightClipper)
                

            train_individual_signal = stats.zscore(train_individual_signal, axis=1)
            train_individual_signal = np.expand_dims(train_individual_signal, axis=1)  
    

            if emotion_name == 'valence':
                train_label = train_individual_label[:,0]
            else: # arousal
                train_label = train_individual_label[:,1] 


            scaled_train_label = utils.min_max_scale(train_label)
            scaled_train_label = np.expand_dims(scaled_train_label,   axis=1)

            train_dataset = utils.load_dataset_to_device(train_individual_signal, scaled_train_label,  batch_size=64, shuffle_flag=True)

            Net, _ = pre_train(Net, train_dataset)



            '''Use the pre-trained Net for each session of the same Subject'''

            '''Test Start'''
            for file_name in test_file_list:    
                test_individual_signal   = np.load(test_signal_path    + file_name)
                test_individual_label    = np.load(test_label_path     + file_name)

                test_individual_signal   = stats.zscore(test_individual_signal, axis=1)
                
                
                if emotion_name == 'valence':
                    test_label   = test_individual_label[:,0]
                else: # arousal
                    test_label   = test_individual_label[:,1] 

                test_individual_signal = np.expand_dims(test_individual_signal, axis=1)
    
                scaled_test_label   = utils.min_max_scale(test_label)
                scaled_test_label   =   np.expand_dims(scaled_test_label,   axis=1)
            
                test_dataset  = utils.load_dataset_to_device(test_individual_signal,   scaled_test_label,    batch_size=64, shuffle_flag=False)


                pred = post_test(Net, test_dataset)

                pred = pred.flatten().tolist()

                # utils.set_permissions_recursive(result_path, 0o2700)

                # Write result
        
                if os.path.exists(result_path + file_name.replace('.npy', '.csv')):
                    try:
                        df = pd.read_csv(result_path + file_name.replace('.npy', '.csv'))
                        pass    
                    except pd.errors.EmptyDataError:
                        shutil.copy(final_label_path + file_name.replace('.npy', '.csv'), result_path + file_name.replace('.npy', '.csv'))
                                
                else:
                    shutil.copy(final_label_path + file_name.replace('.npy', '.csv'), result_path + file_name.replace('.npy', '.csv'))

                utils.write_result(emotion_name, pred, result_path + file_name.replace('.npy', '.csv'))


    else:
        pass




















