import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import math
import os


parent_path = '/path/to/folder'

def Average(lst):
    return sum(lst) / len(lst)

def to_categorical(y):
    """ 1-hot encodes a tensor """
    num_classes = len(np.unique(y))
    return np.eye(num_classes, dtype='uint8')[y.astype(int)]


class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)

class WeightInit(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        torch.manual_seed(0)
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = nn.init.normal_(w, 0.0, 0.02)



def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr





def load_dataset_to_device(data, label, batch_size, shuffle_flag=True):


    data, label = torch.Tensor(data), torch.Tensor(label)

    dataset = torch.utils.data.TensorDataset(data, label)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=shuffle_flag, drop_last=False, pin_memory=True)

    return dataset


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]



# def min_max_scale(label):
    
#     return np.asarray([(x -np.min(label)) / (np.max(label) - np.min(label)) for x in label])


def min_max_scale(label):
    
    return np.asarray([(x - 0.5) / (9.5 - 0.5) for x in label])


def min_max_inverse_scale(scaled_label):
    
    return np.asarray([x  * (9.5 - 0.5) + 0.5 for x in scaled_label])




def rmse(y_pred, y_true):
    mse = np.mean((y_pred - y_true)**2)
    rmse = np.sqrt(mse)
    return rmse


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass



# check format and details of final annotations files 




def write_result(emotion_name, prediction_label, destination_file):
    
    destination = pd.read_csv(destination_file)
    destination = np.array(destination)
    print(len(prediction_label))

    assert len(prediction_label) == len(destination)

    if emotion_name == 'valence': 
        destination[:, 1] = prediction_label
    else:
        destination[:, 2] = prediction_label

    updated_df = pd.DataFrame(destination, columns=['time', 'valence', 'arousal'])
    updated_df.to_csv(destination_file, index=False)




def set_permissions_recursive(path):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o600)
        for f in files:
            os.chmod(os.path.join(root, f), 0o700)



