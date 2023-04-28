import pandas as pd
import os 
import numpy as np
from tqdm import tqdm
import parsing
import utils
import random
import shutil


parser = parsing.create_parser()
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}



scenario_num = args.scenario
fold_num = args.fold

parent_path = utils.parent_path + '/EPiC-2023-competition'



random.seed(123)




def load_data(scenario_num, fold_num=None):

    if scenario_num==1:    
        train_path  = parent_path + '/scenario_{}'.format(scenario_num) + '/train'
        test_path   = parent_path + '/scenario_{}'.format(scenario_num) + '/test'

    else: 
        train_path  = parent_path + '/scenario_{}/fold_{}'.format(scenario_num, fold_num) + '/train'
        test_path  = parent_path + '/scenario_{}/fold_{}'.format(scenario_num, fold_num) + '/test'

    train_data_path  = train_path + '/physiology/'
    train_label_path = train_path + '/labels/'


    test_data_path  = test_path + '/physiology/'
    test_label_path = test_path + '/labels/'

    utils.make_dir(train_label_path)
    utils.make_dir(test_label_path)


    return train_path, test_path, train_data_path, train_label_path, test_data_path, test_label_path





def train_test_files_remove_headers(train_path, test_path, train_data_path, train_label_path, test_data_path, test_label_path):

    # Remove file headers 

    files = os.listdir(train_data_path)
    for file_name in files:           
        if pd.read_csv((train_data_path + file_name),  index_col=0).index.name == 'time' :  

            raw_data    = pd.read_csv((train_data_path              +  file_name), index_col="time") 
            # annotations starting from 0 second should correspond to data collected before 0 second, so need to remove the first one
            annotations = pd.read_csv((train_path + '/annotations/' + file_name), index_col="time")[1:]

            np.savetxt(train_data_path + file_name,   raw_data,      delimiter=',') 
            np.savetxt(train_label_path + file_name,  annotations,   delimiter=',') 


    files = os.listdir(test_data_path)
    for file_name in files:          
        if pd.read_csv((test_data_path +  file_name), index_col=0).index.name == 'time':  

            raw_data    = pd.read_csv((test_data_path               +  file_name), index_col="time") 
            annotations = pd.read_csv((test_path + '/annotations/'  + file_name), index_col="time") # Do NOT touch test since first and last 10 seconds removed

            np.savetxt(test_data_path + file_name,   raw_data,      delimiter=',') 
            np.savetxt(test_label_path + file_name,  annotations,   delimiter=',') 

    return 



def main():
    
    if scenario_num ==1:
        train_path, test_path, train_data_path, train_label_path, val_data_path, val_label_path = load_data(scenario_num)
    else: 
        train_path, test_path, train_data_path, train_label_path, val_data_path, val_label_path = load_data(scenario_num, fold_num)

    train_test_files_remove_headers(train_path, test_path, train_data_path, train_label_path, val_data_path, val_label_path)




if __name__ == "__main__":
    # Code to be executed only when the file is run as the main program
    main()