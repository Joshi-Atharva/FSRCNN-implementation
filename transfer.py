
import torch
from torch import nn
# aliter: import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
import random

import numpy as np
import cv2
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
from collections import namedtuple
from torchvision import models

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transfer model
def transfer_model(PATH, dataset, model, optimizer, epoch_loss_list, epoch_acc_list, train_size, test_size, prev_epochs = 0):
    checkpnt_flag = 1
    # transfering previous checkpoint
    try:
    #{
        checkpoint = torch.load(PATH, weights_only = True)
        print('checkpoint loaded successfully')
        transfer = int(input("transfer previous model? 1/0: "))
        if(transfer == 1):
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_loss_list.extend(checkpoint['epoch_loss_list'])
            epoch_acc_list.extend(checkpoint['epoch_acc_list'])
            prev_epochs = checkpoint['epoch']
            print('model transfered successfully')
        
        # this utility is throwing error - can't pickle the train_set and test_set instances. 
        # For now, I have fixed the seed value to ensure that always the same partition is being done
        # plan to work it out later
        transfer_sets = int(input('transfer the previous train, val, and test sets? (1/0): '))
        if(transfer_sets == 1):
        #{
            try:
                train_set = checkpoint['train_set']
                # val_set = checkpoint['val_set']
                test_set = checkpoint['test_set']
                print('partitioned datasets loaded successfully')
            except:
                print('couldn\'t load partitioned datasets. gotta partition them freshly')
                # torch.manual_seed(5)
                # train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
                train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        #}
        else:
            # train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
            train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        
    #}
    except Exception as e:
        print('Exception occured, running without loading checkpoint')
        checkpnt_flag = 0
        print(e)
    
    if checkpnt_flag == 0:
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    return train_set, test_set, prev_epochs # returning prev_epochs is important as in python there is no straightforward provision to modify integers inside functions (passing by reference)