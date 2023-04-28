import numpy as np
import math
import random
from math import log, e
from tqdm import tqdm
from scipy.stats import *
import copy
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os,sys,inspect
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.preprocessing import MinMaxScaler
import argparse
import utils
import ecg_net
import train_val_split
from scipy import stats
import matplotlib.pyplot as plt


import parsing

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


def eval_step(inputs, labels, model):


    outputs  = model(inputs)

    loss = nn.MSELoss()(outputs, labels)

    outputs = outputs.detach().cpu().clone().numpy()
    labels  = labels.detach().cpu().clone().numpy()


    return loss.item(), outputs, labels



def train(Net, train_dataset, test_dataset):


    train_loss_epoch   = np.ones((args.epochs,  1))
    test_loss_epoch    = np.ones((args.epochs,  1))




    '''
    ***Start of Training***
    '''

    for epoch in range(args.epochs):
        start = time.time()
        train_loss_batch = []
        test_loss_batch = []
        test_pred_batch  = []
        test_truth_batch = []



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

            for data_batch, label_batch in test_dataset:

                data_batch, label_batch =  map(lambda x: x.to(device), (data_batch, label_batch))

                loss, pred, truth = eval_step(data_batch, label_batch, Net)

                test_loss_batch.append(loss)
                test_pred_batch.append(pred)
                test_truth_batch.append(truth)



        test_truth_batch = [item for sublist in test_truth_batch for item in sublist]
        test_pred_batch = [item for sublist in test_pred_batch for item in sublist]


        # plt.plot(np.asarray(test_pred_batch), label='pred')
        # plt.plot(np.asarray(test_truth_batch))
        # plt.legend()
        # plt.show()

        test_loss_epoch[epoch] = utils.Average(test_loss_batch)
        
        
        print('epoch: {}'.format(epoch), 'train_mse_loss: {}'.format(train_loss_epoch[epoch]), 'test_mse_loss: {}'.format(test_loss_epoch[epoch]))

    #     if test_loss_epoch[epoch] <= np.min(test_loss_epoch):

    #         pred_arr, truth_arr = np.squeeze(np.asarray(test_pred_batch)), np.squeeze(np.asarray(test_truth_batch))

 
    test_pred_batch  = utils.min_max_inverse_scale(test_pred_batch)
    test_truth_batch = utils.min_max_inverse_scale(test_truth_batch)
 
    # print(len(test_truth_batch), test_truth_batch[0:2], test_truth_batch[-1])
    # exit(0)
    
    rmse = utils.rmse(np.array(test_pred_batch), np.array(test_truth_batch))
    # plt.plot(test_truth_batch, label='truth')
    # plt.legend()
    # plt.show()
    
    return Net, rmse


if __name__ == '__main__':



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # results save path, depends on method and the arguments in parsing


    #  
    signal_name = args.modality
    emotion_name = args.emotion

    scenario_num = args.scenario
    fold_num = args.fold


    if scenario_num ==1: 
        result_path = utils.parent_path + '/EPiC_val_result' + '/scenario_{}/'.format(scenario_num)
    else: 
        result_path = utils.parent_path + '/EPiC_val_result' + '/scenario_{}/fold_{}/'.format(scenario_num, fold_num)

    
    utils.make_dir(result_path)


    if scenario_num == 1: 
        train_path, val_path, _, _, _, _ = train_val_split.make_train_val(scenario_num)
    else: 
        train_path, val_path, _, _, _, _ = train_val_split.make_train_val(scenario_num, fold_num)


    '''
    Indepdendent training , do not change this part ------START
    '''


    if args.pretraining ==True:

        train_entire_signal = np.load(train_path + '/{}/'.format(signal_name)          + 'sliding_window/' + 'entire.npy')
        val_entire_signal   = np.load(val_path   + '/{}/'.format(signal_name)          + 'sliding_window/' + 'entire.npy')
        train_entire_label  = np.load(train_path + '/labels/'  + 'sliding_window/' + 'entire.npy')
        val_entire_label    = np.load(val_path   + '/labels/'  + 'sliding_window/' + 'entire.npy')

        # print(train_entire_ecg.shape, val_entire_ecg.shape, train_entire_label.shape, val_entire_label.shape)



        train_entire_signal = stats.zscore(train_entire_signal, axis=1)
        val_entire_signal   = stats.zscore(val_entire_signal, axis=1)




        # emotion_label = ['valence', 'arousal']
        if emotion_name == 'valence':
            train_label = train_entire_label[:,0]
            val_label   = val_entire_label[:,0]

        else: # arousal
            train_label = train_entire_label[:,1] 
            val_label   = val_entire_label[:,1] 


        scaled_train_label = utils.min_max_scale(train_label)
        scaled_val_label   = utils.min_max_scale(val_label)

        # scaled_back_train_label = utils.min_max_inverse_scale(train_label, scaled_train_label)



        Net = ecg_net.Conv_EEG().to(device)
           
        Net.apply(utils.WeightInit)
        Net.apply(utils.WeightClipper)
        print(Net)

        train_entire_signal, val_entire_signal = np.expand_dims(train_entire_signal, axis=1),np.expand_dims(val_entire_signal, axis=1)
        scaled_train_label, scaled_val_label   = np.expand_dims(scaled_train_label, axis=1), np.expand_dims(scaled_val_label, axis=1)


        train_dataset = utils.load_dataset_to_device(train_entire_signal, scaled_train_label,  batch_size=64,  shuffle_flag=True)
        test_dataset  = utils.load_dataset_to_device(val_entire_signal,   scaled_val_label,    batch_size=64, shuffle_flag=False)

        
        Net,_ = train(Net, train_dataset, test_dataset)


        ckpt_path = train_path + '/{}/'.format(signal_name) + 'sliding_window/' + '{}_'.format(emotion_name) + 'model' 


        torch.save(Net.state_dict(), ckpt_path)

    
    else: 
        
        '''
        Subject and session depdendent training , do not change this part ------START
        '''

        train_signal_path   = train_path + '/{}/'.format(signal_name) + 'sliding_window/'
        val_signal_path     = val_path + '/{}/'.format(signal_name) + 'sliding_window/'

        train_label_path    = train_path + '/labels/' + 'sliding_window/'
        val_label_path      = val_path + '/labels/' + 'sliding_window/'



        files = os.listdir(train_label_path)
        files.remove('entire.npy')



        if scenario_num ==1:

            rmse_list=[]
            for file_name in tqdm(files):


                Net = ecg_net.Conv_EEG().to(device)
                if args.use_pretrain == True: 
                    ckpt_path = train_path + '/{}/'.format(signal_name) + 'sliding_window/' + '{}_'.format(emotion_name) + 'model' 
                    Net.load_state_dict(torch.load(ckpt_path))
                else: 
                    Net.apply(utils.WeightInit)
                    Net.apply(utils.WeightClipper)
                    


                train_individual_signal = np.load(train_signal_path  + file_name)
                val_individual_signal   = np.load(val_signal_path    + file_name)

                train_individual_label  = np.load(train_label_path   + file_name)
                val_individual_label    = np.load(val_label_path     + file_name)



                train_individual_signal = stats.zscore(train_individual_signal, axis=1)
                val_individual_signal   = stats.zscore(val_individual_signal, axis=1)

                # plt.plot(train_individual_signal[2])
                # plt.plot(val_individual_signal[1])

                # plt.show()

                # emotion_label = ['valence', 'arousal']
                if emotion_name == 'valence':
                    train_label = train_individual_label[:,0]
                    val_label   = val_individual_label[:,0]

                else: # arousal
                    train_label = train_individual_label[:,1] 
                    val_label   = val_individual_label[:,1] 

                        
                # plt.plot(val_label,label='original')   
                # plt.legend()
                # plt.show()

                scaled_train_label = utils.min_max_scale(train_label)
                scaled_val_label   = utils.min_max_scale(val_label)

                # print(len(val_label), val_label[0:2], val_label[-1])
                
                
                train_individual_signal, val_individual_signal  = np.expand_dims(train_individual_signal, axis=1),   np.expand_dims(val_individual_signal, axis=1)
                scaled_train_label, scaled_val_label      = np.expand_dims(scaled_train_label,   axis=1),   np.expand_dims(scaled_val_label,   axis=1)

                # print(file_name, len(scaled_val_label), scaled_val_label[3:5]) 

                train_dataset = utils.load_dataset_to_device(train_individual_signal, scaled_train_label,  batch_size=16, shuffle_flag=True)
                test_dataset  = utils.load_dataset_to_device(val_individual_signal,   scaled_val_label,    batch_size=16, shuffle_flag=False)


                
                Net, rmse = train(Net, train_dataset, test_dataset)
                
                rmse_list.append(rmse)
            
                # print(rmse)

            mse_list = np.asarray(rmse_list)
            
            
            if args.use_pretrain == True: 
                result_path = result_path + 'retrain/'
            
            utils.make_dir(result_path)

            np.savetxt(result_path+'{}_{}.csv'.format(signal_name, emotion_name), mse_list, delimiter=',')





        elif scenario_num ==2: 
            # Session dependent 
            session_list = [0, 2, 9, 10, 11, 13, 14, 20]


            train_files = os.listdir(train_label_path)
            train_files.remove('entire.npy')
            # train_sub_list =utils.unique([file_name.split('_')[1] for file_name in train_files])
        
            val_files = os.listdir(val_label_path)
            val_files.remove('entire.npy')
            val_sub_list =utils.unique([file_name.split('_')[1] for file_name in val_files])


            rmse_list=[]

            for session_num in session_list:
                # print(train_files)
                
                train_individual_signal = np.zeros((0, 2000)) # 10s window * downsampled 200Hz data
                train_individual_label = np.zeros((0, 2))
                val_individual_signal = np.zeros((0, 2000)) # 10s window * downsampled 200Hz data
                val_individual_label = np.zeros((0, 2))
                

                train_file_list = [x for x in train_files if 'vid_{}.'.format(session_num) in x]
                val_file_list   = [x for x in val_files if 'vid_{}.'.format(session_num) in x]
                
                for file_name in train_file_list:
                    train_individual_signal = np.vstack((train_individual_signal, np.load(train_signal_path  + file_name)))
                    train_individual_label  = np.vstack((train_individual_label,  np.load(train_label_path   + file_name)))


                for file_name in val_file_list:    
                    val_individual_signal   = np.vstack((val_individual_signal, np.load(val_signal_path    + file_name)))
                    val_individual_label    = np.vstack((val_individual_label, np.load(val_label_path     + file_name)))


                print(train_individual_signal.shape,val_individual_label.shape)
                

                Net = ecg_net.Conv_EEG().to(device)
                if args.use_pretrain == True: 
                    ckpt_path = train_path + '/{}/'.format(signal_name) + 'sliding_window/' + '{}_'.format(emotion_name) + 'model' 
                    Net.load_state_dict(torch.load(ckpt_path))
                else: 
                    Net.apply(utils.WeightInit)
                    Net.apply(utils.WeightClipper)
                    

                train_individual_signal = stats.zscore(train_individual_signal, axis=1)
                val_individual_signal   = stats.zscore(val_individual_signal, axis=1)


                if emotion_name == 'valence':
                    train_label = train_individual_label[:,0]
                    val_label   = val_individual_label[:,0]

                else: # arousal
                    train_label = train_individual_label[:,1] 
                    val_label   = val_individual_label[:,1] 


                scaled_train_label = utils.min_max_scale(train_label)
                scaled_val_label   = utils.min_max_scale(val_label)


                train_individual_signal, val_individual_signal  = np.expand_dims(train_individual_signal, axis=1),   np.expand_dims(val_individual_signal, axis=1)
                scaled_train_label, scaled_val_label      = np.expand_dims(scaled_train_label,   axis=1),   np.expand_dims(scaled_val_label,   axis=1)


                train_dataset = utils.load_dataset_to_device(train_individual_signal, scaled_train_label,  batch_size=64, shuffle_flag=True)
                test_dataset  = utils.load_dataset_to_device(val_individual_signal,   scaled_val_label,    batch_size=64, shuffle_flag=False)


                
                Net, rmse = train(Net, train_dataset, test_dataset)
                
                rmse_list.append(rmse)
            
                # print(rmse)

            mse_list = np.asarray(rmse_list)
                
                
            if args.use_pretrain == True: 
                result_path = result_path +'retrain/'
            
            utils.make_dir(result_path)

            np.savetxt(result_path+'{}_{}.csv'.format(signal_name, emotion_name), mse_list, delimiter=',')




        elif scenario_num in [3,4]: 
            # Subject dependent 
        
            train_files = os.listdir(train_label_path)
            train_files.remove('entire.npy')
            val_files   = os.listdir(val_label_path)
            val_files.remove('entire.npy')
            sub_list =utils.unique([file_name.replace('.csv', '').split('_')[1] for file_name in train_files])

            print(len(sub_list))


            rmse_list=[]

            for sub in sub_list:
                # print(train_files)
                
                train_individual_signal = np.zeros((0, 2000)) # 10s window * downsampled 200Hz data
                train_individual_label = np.zeros((0, 2))
                val_individual_signal = np.zeros((0, 2000)) # 10s window * downsampled 200Hz data
                val_individual_label = np.zeros((0, 2))


                train_file_list = [x for x in train_files if 'sub_' + sub + '_' in x]
                val_file_list   = [x for x in val_files if 'sub_' + sub + '_'in x]

            
                for file_name in train_file_list:
                    train_individual_signal = np.vstack((train_individual_signal, np.load(train_signal_path  + file_name)))
                    train_individual_label  = np.vstack((train_individual_label,  np.load(train_label_path   + file_name)))


                for file_name in val_file_list:    
                    val_individual_signal   = np.vstack((val_individual_signal, np.load(val_signal_path    + file_name)))
                    val_individual_label    = np.vstack((val_individual_label, np.load(val_label_path     + file_name)))

                print(train_individual_signal.shape,val_individual_label.shape)

                Net = ecg_net.Conv_EEG().to(device)
                if args.use_pretrain == True: 
                    ckpt_path = train_path + '/{}/'.format(signal_name) + 'sliding_window/' + '{}_'.format(emotion_name) + 'model' 
                    Net.load_state_dict(torch.load(ckpt_path))
                else: 
                    Net.apply(utils.WeightInit)
                    Net.apply(utils.WeightClipper)
                    

                train_individual_signal = stats.zscore(train_individual_signal, axis=1)
                val_individual_signal   = stats.zscore(val_individual_signal, axis=1)


                if emotion_name == 'valence':
                    train_label = train_individual_label[:,0]
                    val_label   = val_individual_label[:,0]

                else: # arousal
                    train_label = train_individual_label[:,1] 
                    val_label   = val_individual_label[:,1] 


                scaled_train_label = utils.min_max_scale(train_label)
                scaled_val_label   = utils.min_max_scale(val_label)


                train_individual_signal, val_individual_signal  = np.expand_dims(train_individual_signal, axis=1),   np.expand_dims(val_individual_signal, axis=1)
                scaled_train_label, scaled_val_label      = np.expand_dims(scaled_train_label,   axis=1),   np.expand_dims(scaled_val_label,   axis=1)


                train_dataset = utils.load_dataset_to_device(train_individual_signal, scaled_train_label,  batch_size=64, shuffle_flag=True)
                test_dataset  = utils.load_dataset_to_device(val_individual_signal,   scaled_val_label,    batch_size=64, shuffle_flag=False)


                
                Net, rmse = train(Net, train_dataset, test_dataset)
                
                rmse_list.append(rmse)
            
                # print(rmse)

            mse_list = np.asarray(rmse_list)
                
                
            if args.use_pretrain == True: 
                result_path = result_path +'retrain/'
            
            utils.make_dir(result_path)

            np.savetxt(result_path+'{}_{}.csv'.format(signal_name, emotion_name), mse_list, delimiter=',')


        else:
            print('Please select correct senario number from 1 to 4!')