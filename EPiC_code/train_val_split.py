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

parent_path = utils.parent_path + '/EPiC-2023-competition'


random.seed(123)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

    return 



def load_original_train(scenario_num, fold_num=None):

    if scenario_num==1:    
        train_path  = parent_path + '/scenario_{}'.format(scenario_num) + '/train'
    else: 
        train_path  = parent_path + '/scenario_{}/fold_{}'.format(scenario_num, fold_num) + '/train'

    original_train_data_path  = train_path + '/physiology/'
    original_train_label_path = train_path + '/annotations/'

    return original_train_data_path, original_train_label_path


def make_train_val(scenario_num, fold_num=None):
    if scenario_num==1:    

        train_path  = parent_path + '/scenario_{}'.format(scenario_num) + '/split/train'
        val_path    = parent_path + '/scenario_{}'.format(scenario_num) + '/split/val'
    
    else: 
        train_path  = parent_path + '/scenario_{}/fold_{}'.format(scenario_num, fold_num) + '/split/train'
        val_path    = parent_path + '/scenario_{}/fold_{}'.format(scenario_num, fold_num) + '/split/val'


    train_label_path = train_path + '/labels/'
    val_label_path   = val_path   + '/labels/'
    train_data_path  = train_path + '/raw_data/'
    val_data_path    = val_path   + '/raw_data/'

    for path in [train_label_path, val_label_path, train_data_path, val_data_path]:
        make_dir(path)
        
    return train_path, val_path, train_label_path, val_label_path, train_data_path, val_data_path


def train_val_files_copy(files, train_string, original_train_data_path, original_train_label_path, train_data_path, train_label_path, val_data_path, val_label_path):


    for file_name in files:
        if any(string in file_name for string in train_string):
            shutil.copy(original_train_data_path  + file_name, train_data_path  + file_name)
            shutil.copy(original_train_label_path + file_name, train_label_path + file_name)

        else: 
            shutil.copy(original_train_data_path  + file_name, val_data_path    + file_name)
            shutil.copy(original_train_label_path + file_name, val_label_path   + file_name)

    return 

def train_val_files_remove_headers(train_data_path, train_label_path, val_data_path, val_label_path):

    # Remove file headers 

    data_path_list  = [train_data_path, val_data_path]
    label_path_list = [train_label_path, val_label_path]


    for data_path, label_path in zip(data_path_list, label_path_list):

        files = os.listdir(data_path)
        for file_name in files:           
            if pd.read_csv((data_path +  file_name),  index_col=0).index.name == 'time':  

                raw_data    = pd.read_csv((data_path +  file_name), index_col="time")
                np.savetxt(data_path + file_name,   raw_data,      delimiter=',') 

            if pd.read_csv((label_path +  file_name),  index_col=0).index.name == 'time':  
                # annotations starting from 0 second should correspond to data collected before 0 second, so need to remove the first one
                annotations = pd.read_csv((label_path + file_name), index_col="time")[1:]
                np.savetxt(label_path + file_name,  annotations,   delimiter=',') 
                
            else:
                pass
    return 



def main():
    
    if scenario_num ==1: 

        original_train_data_path, original_train_label_path = load_original_train(scenario_num)
        _, _, train_label_path, val_label_path, train_data_path, val_data_path = make_train_val(scenario_num)

        
        files = os.listdir(original_train_data_path)

        for file_name in tqdm(files):
            if pd.read_csv((original_train_data_path +  file_name),  index_col=0).index.name == 'time':                
                raw_data    = pd.read_csv((original_train_data_path +  file_name), index_col="time") 
            else: 
                raw_data    = pd.read_csv((original_train_data_path +  file_name))
                
            if pd.read_csv((original_train_label_path +  file_name),  index_col=0).index.name == 'time':  
                annotations = pd.read_csv((original_train_label_path + file_name), index_col="time")[1:]
            else:
                annotations = pd.read_csv((original_train_label_path + file_name))

            # split annotations to train/val

            train_label_file = train_label_path + file_name
            val_label_file   = val_label_path   + file_name


            # split to half
            label_1_half_len = int(annotations.shape[0]//2)

            annotations_train = annotations[:label_1_half_len]
            annotations_val   = annotations[label_1_half_len:]

            np.savetxt(train_label_file, annotations_train, delimiter=',') 
            np.savetxt(val_label_file,   annotations_val,   delimiter=',') 


            # split raw data to train/val

            data_1_half_len  = label_1_half_len * 50
            data_full_len    = annotations.shape[0]*50

            train_data_file = train_data_path + file_name
            val_data_file   = val_data_path   + file_name

            data_train = raw_data[ : data_1_half_len]
            val_train  = raw_data[data_1_half_len :data_full_len]


            assert(data_train.shape[0]+val_train.shape[0] <= raw_data.shape[0])


            np.savetxt(train_data_file, data_train, delimiter=',')  
            np.savetxt(val_data_file, val_train, delimiter=',') 



    elif scenario_num ==2: 

        for fold_num in tqdm(range(5)):

            original_train_data_path, original_train_label_path = load_original_train(scenario_num, fold_num)
            _, _, train_label_path, val_label_path, train_data_path, val_data_path = make_train_val(scenario_num, fold_num)
             
            files = os.listdir(original_train_data_path)
            
            # define an empty list to store subject numbers
            sub_num = []

            # iterate through each filename
            for file_name in files:
                # split the filename by '_' character
                parts = file_name.split('_')
          
                # append the subject number to the list
                sub_num.append(parts[1])

            # print the list of subject numbers
            sub_list = utils.unique(sub_num)

            # 80-20 train/val split with fixed random_seeds
            train_portion = int(len(sub_list) * 0.8)
       
            train_list = random.sample(sub_list, train_portion)

            # val_list =  [x for x in sub_list if x not in train_list]

            train_string = ['sub_' + x + '_' for x in train_list]
            # val_string   = ['sub_' + x for x in val_list]
            # print(train_string)

            # move files  
            train_val_files_copy(files, train_string, original_train_data_path, original_train_label_path, train_data_path, train_label_path, val_data_path, val_label_path)

            # Remove file headers 

            train_val_files_remove_headers(train_data_path, train_label_path, val_data_path, val_label_path)




    elif scenario_num ==3: # Across elicitor valiadation 
        # Observe per elicitor video list [[16, 20], [0, 3], [10, 22], [4, 21]] per 30 subjects
        
        
        for fold_num in tqdm(range(4)):
        
            elicit_per_fold = [[16, 20], [0, 3], [10, 22], [4, 21]] 
            original_train_data_path, original_train_label_path = load_original_train(scenario_num, fold_num)
            _, _, train_label_path, val_label_path, train_data_path, val_data_path = make_train_val(scenario_num, fold_num)
             
            files = os.listdir(original_train_data_path) 
            
            elicit_per_fold.pop(fold_num) # pop out the test elicitor 


            train_elicitor_list = random.sample(elicit_per_fold, 2) # Randomly select two sublists out of three sublists


            print(train_elicitor_list)
            train_list = [element for sublist in train_elicitor_list for element in sublist]

            # select video list for training 
            train_string = ['vid_{}.csv'.format(x) for x in train_list]
            print(train_string)


            # move files  
            train_val_files_copy(files, train_string, original_train_data_path, original_train_label_path, train_data_path, train_label_path, val_data_path, val_label_path)

            # Remove file headers 

            train_val_files_remove_headers(train_data_path, train_label_path, val_data_path, val_label_path)


    elif scenario_num ==4: # Across version valiadation 
        # Observe per version video list in train fold: [[3, 16, 19, 20], [0, 9, 12, 15]]
        
        version_per_fold = [[3, 16, 19, 20], [0, 9, 12, 15]]

        for fold_num in tqdm(range(2)):


            original_train_data_path, original_train_label_path = load_original_train(scenario_num, fold_num)
            _, _, train_label_path, val_label_path, train_data_path, val_data_path = make_train_val(scenario_num, fold_num)


            files = os.listdir(original_train_data_path) 

    
            train_version_list = random.sample(version_per_fold[fold_num], 2) # Randomly select two elements out of four


            # select video list for training 
            train_string = ['vid_{}.csv'.format(x) for x in train_version_list]
            # print(train_vid_string)


            # move files  
            train_val_files_copy(files, train_string, original_train_data_path, original_train_label_path, train_data_path, train_label_path, val_data_path, val_label_path)
                
            # Remove file headers 

            train_val_files_remove_headers(train_data_path, train_label_path, val_data_path, val_label_path)


    else: 
        AssertionError('Please select valid scenario number')







if __name__ == "__main__":
    # Code to be executed only when the file is run as the main program
    main()