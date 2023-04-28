import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np
import torch as nn 
import preprocessing
from tqdm import tqdm
import utils
import train_val_split, train_test_split
import parsing
parser = parsing.create_parser()
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


scenario_num = args.scenario
fold_num = args.fold

signal_data_index = {'ecg': 0 ,'bvp': 1, 'gsr': 2, 'rsp': 3, 'skt': 4, 'emg_zygo': 5, 'emg_coru': 6, 'emg_trap': 7}
signal_window_size = {'ecg': 10 ,'bvp': 10, 'gsr': 10, 'rsp': 10, 'skt': 10, 'emg_zygo': 0.5, 'emg_coru': 0.5, 'emg_trap': 0.5}


if args.final_flag ==True:
    if scenario_num ==1:
        train_path, val_path, _, _, _, _ = train_test_split.load_data(scenario_num)
    else: 
        train_path, val_path, _, _, _, _ = train_test_split.load_data(scenario_num, fold_num)

else: 
    if scenario_num ==1: 
        train_path, val_path, _, _, _, _ = train_val_split.make_train_val(scenario_num)
    else: 
        train_path, val_path, _, _, _, _ = train_val_split.make_train_val(scenario_num, fold_num)



def save_filtered_data(signal_name):

    for path in [train_path, val_path]:

        save_signal_path  = path + '/{}/'.format(signal_name)
        
        
        if not os.path.exists(save_signal_path):
            os.makedirs(save_signal_path)

        if args.final_flag ==True: # in final test, not necessay to copy from folder 'physiology' to 'raw_data'
            files = os.listdir(path + '/physiology/')
        else:
            files = os.listdir(path + '/raw_data/')
    
        # print the name of each file
        for file_name in tqdm(files):
            # print(path + '/raw_data/' + file_name)\
            if args.final_flag ==True: 
                raw_data = pd.read_csv(path + '/physiology/' + file_name, header=None)
            else:
                raw_data = pd.read_csv(path + '/raw_data/' + file_name, header=None)
            signal = np.asarray(raw_data)[:,signal_data_index[signal_name]] # read single-modality only (e.g., ecg, ppg)

            if signal_name == 'ecg':
               filtered_data = preprocessing.FILTER_ECG()(signal)
            elif signal_name== 'gsr':
               filtered_data = preprocessing.FILTER_GSR()(signal)

                
            np.savetxt(save_signal_path + file_name, filtered_data, delimiter=',')
            




def make_sliding_window(data, label, win_size_sec):
    fs = 200
    win_size = win_size_sec * fs # ecg, gsr downsampled to 200Hz, conver to data points
    
    if len(data)//10 >= len(label):
        data = data[:len(label)*10]
    else: 
        # print('report')
        label = label[:-1]
    
    data_win_len  = (len(data) - win_size)//10 + 1 # to align with annotations. ratio = 200Hz/ 20Hz
    label_win_len = len(label) - win_size_sec * 20 + 1



    assert(data_win_len == label_win_len)

    new_data = np.zeros((data_win_len, win_size))


    move_point = fs//20 # 200Hz data, 20 Hz annotations 

    for i in range(data_win_len):
        new_data[i] = data.flatten()[i*move_point: i*move_point + win_size]
    


    new_label = label[win_size_sec*20-1:]

    return new_data, new_label


def make_sliding_window_test_only(data, label, win_size_sec):
    fs = 200
    win_size = win_size_sec * fs # ecg, gsr downsampled to 200Hz, conver to data points
    
    # Test label files already cutoff first and last 10 seconds 
    
    if len(data) >= len(label)*10 + 2*win_size:
        data = data[:len(label)*10 + 2*win_size]
    else:
        pass
        # print(',,,,,,,,,,,,,,,,,,', len(data))
      


    data_win_len  = (len(data) - 2*win_size)//10 +1  # to align with annotations. ratio = 200Hz/ 20Hz

    label_win_len = len(label) # Already cutoff first and last 10 secs

    # print(data_win_len, label_win_len)

    assert(data_win_len == label_win_len)

    new_data = np.zeros((data_win_len, win_size))

    # print(new_data.shape)

    move_point = fs//20 # 200Hz data, 20 Hz annotations 

    for i in range(data_win_len):
        new_data[i] = data.flatten()[i*move_point: i*move_point + win_size]
    
    new_label = label

    return new_data, new_label




window_size = 10  # in seconds

def load_filtered_data(signal_name):
    for path in [train_path, val_path]:

        load_signal_path  = path + '/{}/'.format(signal_name)

        
        label_path = path + '/labels/'

        all_entries = os.listdir(label_path)
        files = [entry for entry in all_entries if os.path.isfile(os.path.join(label_path, entry))]


        save_signal_win_path   = load_signal_path + 'sliding_window/'
        save_label_win_path = label_path + 'sliding_window/'


        utils.make_dir(save_signal_win_path)    
        utils.make_dir(save_label_win_path)


        # print the name of each file
        for file_name in tqdm(files):
   
            filtered_data = pd.read_csv(load_signal_path + file_name, header=None)

            annotations   = pd.read_csv(path + '/labels/' + file_name, header=None)

            filtered_data = np.asarray(filtered_data)
            annotations   = np.asarray(annotations)
            
            # print(file_name)

            if args.final_flag ==True and path==val_path:  
                new_data, new_label = make_sliding_window_test_only(filtered_data, annotations, window_size)
            else:
                new_data, new_label = make_sliding_window(filtered_data, annotations, window_size)

            # File will be larger, save in npy format

            np.save(save_signal_win_path + file_name.replace('.csv', ''),   new_data)
            np.save(save_label_win_path + file_name.replace('.csv', ''),   new_label)


def make_entire_data(signal_name):
    # train all signal --> val on all signal
    train_signal_path = train_path + '/{}/'.format(signal_name) + 'sliding_window/'
    val_signal_path   = val_path   + '/{}/'.format(signal_name) + 'sliding_window/'

    train_label_path = train_path + '/labels/' + 'sliding_window/'
    val_label_path   = val_path   + '/labels/' + 'sliding_window/'



    train_entire_signal = np.zeros((0, 2000)) # 10s window * downsampled 200Hz data
    val_entire_signal   = np.zeros((0, 2000))

    train_entire_label = np.zeros((0, 2))
    val_entire_label  =  np.zeros((0, 2)) 


    
    all_entries = os.listdir(train_label_path)
    files = [entry for entry in all_entries if os.path.isfile(os.path.join(train_label_path, entry))]
     

    if 'entire.npy' in files: 
        files.remove('entire.npy')

    for file_name in tqdm(files):
        
        train_entire_signal   = np.vstack((train_entire_signal,   np.load(train_signal_path   + file_name)))
        train_entire_label    = np.vstack((train_entire_label,    np.load(train_label_path    + file_name)))



    all_entries = os.listdir(val_label_path)
    files = [entry for entry in all_entries if os.path.isfile(os.path.join(val_label_path, entry))]
    
    if 'entire.npy' in files: 
        files.remove('entire.npy')

    for file_name in tqdm(files):

        val_entire_label    = np.vstack((val_entire_label,      np.load(val_label_path      + file_name)))
        val_entire_signal   = np.vstack((val_entire_signal,     np.load(val_signal_path     + file_name)))

    
    np.save(train_signal_path    + 'entire',   train_entire_signal)
    np.save(val_signal_path      + 'entire',     val_entire_signal)
    np.save(train_label_path     + 'entire',    train_entire_label)
    np.save(val_label_path       + 'entire',      val_entire_label)






if __name__ == "__main__":
    # Code to be executed only when the file is run as the main program


    signal_name = args.modality

    print(args.scenario, args.fold)

    # Filter Signal
    save_filtered_data(signal_name)

    # load Signal with 10-second window
    load_filtered_data(signal_name)

    # Stack Data
    make_entire_data(signal_name)














