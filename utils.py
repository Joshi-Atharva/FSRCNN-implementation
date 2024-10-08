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
from dataset import SRdatasets

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# defining psnr metric:
class PSNR():
    def single(self, pred, target):

        pred = pred.to(device)
        target = target.to(device)
        while len(pred.shape) < 4:
            pred = pred.unsqueeze(0)
        while len(target.shape) < 4: 
            target = target.unsqueeze(0)

        pred = F.interpolate(pred, size = (480, 320), mode = 'bicubic', align_corners = False)
        
        mse = torch.mean((target - pred) ** 2)
        if mse == 0:
            return 100
        else:
            PIXEL_MAX = 255.0
            mse = mse.item()
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    def batch(self, predictions, target):

        predictions = predictions.to(device)
        target = target.to(device)
        PIXEL_MAX = 255.0

        m = predictions.shape[0]
        mse_acc = 0 # avg mse accumulator
        for i in range(predictions.shape[0]):

            target_inter, predictions_inter = target[i].unsqueeze(0), predictions[i].unsqueeze(0)
            target_inter, predictions_inter = F.interpolate(target_inter, size=(480, 320), mode='bicubic', align_corners=False), F.interpolate(predictions_inter, size=(480, 320), mode='bicubic', align_corners=False)
            # resize required for original LR HR comparison
            mse_acc = mse_acc + torch.mean((target_inter - predictions_inter) ** 2)/m
            
        if mse_acc == 0:
            return 100
        else:
            mse_acc = mse_acc.item()
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse_acc))

# prediction (inference) function
def predict(test_loader):
    batch_psnr = 0;
    avg_psnr = 0;
    predictions_list = []
    for i, (low, high) in enumerate(test_loader):
        low = low.to(device)
        predictions = model(low)
        predictions = predictions.to(device)
        predictions_list.append(predictions.detach().cpu().numpy())
        
        batch_psnr = PSNR().batch(predictions, high)
        avg_psnr = avg_psnr + batch_psnr
    avg_psnr = avg_psnr/len(test_loader)
    try:
        predictions = np.array(predictions_list)
        predictions = torch.Tensor(predictions)
    except:
        # print(predictions_list[0:2])
        predictions = np.array([])
    
    return avg_psnr, predictions

# displaying some randomly picked examples
def display_random(test_loader, model, batch_size = 64):
    print('Some examples: ')
    x = 0 # iterations counter
    for i, (inputs, targets) in enumerate(test_loader):
        for j in np.random.randint(0, batch_size, 10):
            j = min(len(inputs)-1, j) # since last batch might have no. of elements < batch_size if test_size is not a multiple of batch_size
            input_tensor, target_tensor = inputs[j], targets[j]
            input_tensor = input_tensor.unsqueeze(0)
            target_tensor = target_tensor.unsqueeze(0)
            
            input_arr= np.array(input_tensor)
            target = np.array(target_tensor)

            pred_tensor = model(input_tensor.to(device))
            pred = pred_tensor.detach().cpu().numpy()
    
            psnr_input = PSNR().single(pred = input_tensor, target = target_tensor)
            psnr_prediction = PSNR().single(pred = pred_tensor, target = target_tensor)
            if(len(pred.shape) > 3):
                pred = np.squeeze(pred, 0)
            pred = pred.transpose(1, 2, 0)
    
            # use of prediction list - didn't work out and felt unnecessary:
            '''
            prediction = np.array(predictions[i, j])
            prediction = prediction.transpose(1, 2, 0)
            '''
            if(len(input_arr.shape) > 3):
                input_arr = np.squeeze(input_arr, 0)
            if(len(target.shape) > 3):
                target = np.squeeze(target, 0)
            input_arr = input_arr.transpose(1, 2, 0)
            target = target.transpose(1, 2, 0)
            
        
            plt.axis('off')
            
            ax = plt.subplot(1, 3, 1)
            ax.imshow(input_arr.astype('uint8'))
            ax.set_title('input')
            plt.text(0, -100, f'input PSNR: {psnr_input:.4f}')
            ax.axis('off')
        
            ax = plt.subplot(1, 3, 2)
            ax.imshow(pred.astype('uint8'))
            ax.set_title('prediction')
            plt.text(0, -100, f'prediction PSNR: {psnr_prediction:.4f}')
            ax.axis('off')
        
            ax = plt.subplot(1, 3, 3)
            ax.imshow(target.astype('uint8'))
            ax.set_title('target')
            ax.axis('off')
            
            plt.show()
            x = x + 1
            if(x == 10):
                break # prints only 10 images
    
# for observing a particular sample (particular image)
def display_particular(i, model):
    sample_set = SRdatasets()
    input_tensor, target_tensor = sample_set[0] 
    # replace 0 by any desired index of the image in the instance of SRdatasets class
    prediction = model(input_tensor.to(device))
    print('input:')
    plt.imshow(np.array(input_tensor).astype('uint8').transpose(1, 2, 0))
    plt.show()
    print('target:')
    plt.imshow(np.array(target_tensor).astype('uint8').transpose(1, 2, 0))
    plt.show()
    print('prediction:')
    prediction = prediction.detach().cpu().numpy()
    plt.imshow(np.array(prediction).astype('uint8').transpose(1, 2, 0))
    plt.show()
