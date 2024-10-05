import torch
from torch import nn
# aliter: import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

import numpy as np
import cv2
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SRdatasets(Dataset):
    def __init__(self, dataset_path = 'C:/Users/athar/MLprojects/dataloader_task/Datasets', transform = None):
        script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.chdir(dataset_path)
        input_list = []
        target_list = []
        
        for dset_name in os.listdir():
        #{
            os.chdir(dataset_path + '\\' + dset_name)
            for dir_name in os.listdir():
            #{
                dir_num = int(dir_name[-1])

                # changing directory to export images
                cwd = os.getcwd()
                os.chdir(cwd + '\\' + dir_name)
                
                num_images = len(os.listdir())
                num_images = num_images//2
                for i in range(1, num_images):
                    input_arr, target = self.extract("LR", dir_num, i), self.extract("HR", dir_num, i)
                    '''
                    for adding deterministic transforms for data augmentation, add them here as:
                    hflip_input = TF.hflip(input_arr)
                    hflip_target = TF.hflip(target)
                    vflip_input = TF.hflip(input_arr)
                    vflip_target = TF.vflip(target)
                    
                    input_list.extend([input_arr, hflip_input, vflip_input])
                    target_list.extend([target, hflip_target, vflip_target])
                    '''
                    hflip_input = TF.hflip(torch.from_numpy(input_arr))
                    hflip_target = TF.hflip(torch.from_numpy(target))
                    # vflip_input = TF.vflip(torch.from_numpy(input_arr))
                    # vflip_target = TF.vflip(torch.from_numpy(target))
                    
                    input_list.extend([input_arr, hflip_input])
                    target_list.extend([target, hflip_target])
                    
                    # input_list.append(input_arr)
                    # target_list.append(target)
                    
                # returning back to the dset directory
                os.chdir(dataset_path + '\\' + dset_name)
            #}
        #}
        
        # returning back to the current directory 
        # this way the flow of rest of the program is not affected        
        os.chdir(script_directory)
        # converting the list of np.arrays into higher dimenstion np.array since list -> tensor conversion is much slower
        input_arr = np.array(input_list)
        target_arr = np.array(target_list)
        
        self.input_data = torch.Tensor(input_arr)
        self.target_data = torch.Tensor(target_arr)
        self.size = len(self.input_data)
        self.transform = transform
    
        

        
    def extract(self, res = "LR", dir_num = 2, i = 1):
        leading_zeros = 3 - len(str(i))
        number_str = leading_zeros * "0" + str(i)

        final_str = "img_" + number_str + "_SRF_" + str(dir_num) + "_" + res + ".png" 
        img = Image.open(final_str)
        npimg = np.asarray(img) # npimg.shape(480, 320, 3) or (320, 480, 3)

        
        # resizing all images into same dimensions
        npimg = cv2.resize(npimg, dsize = (320, 480), interpolation = cv2.INTER_CUBIC)
        npimg = np.array(npimg)
        if len(npimg.shape) == 2:
            npimg = cv2.cvtColor(npimg, cv2.COLOR_GRAY2BGR) 
        # npimg = npimg.reshape(rows, col, 3)
        return np.transpose(npimg, axes = (2, 0, 1)) # returning in form (num_channels = 3, rows, col)
        
    def __getitem__(self, index):
        if isinstance(index, int):
            input_data, target_data = self.input_data[index], self.target_data[index]

            # obsolete operation but kept for illustration of concept
            # deterministic transform is done in the __init__ method itself
            if random.random() > 1:
                input_data = TF.hflip(input_data)
                target_data = TF.hflip(target_data)
            return input_data, target_data
            
        if isinstance(index, slice):
            if random.random() > 1: # instead of deleting the code I have put an impossible condition
                return torch.stack([TF.hflip(self.input_data[i]) for i in range(*index.indices(len(self)))]), torch.stack([TF.hflip(self.target_data[i]) for i in range(*index.indices(len(self)))])
            else:
                return torch.stack([(self.input_data[i]) for i in range(*index.indices(len(self)))]), torch.stack([(self.target_data[i]) for i in range(*index.indices(len(self)))])
            # return type: tuple of the form: (tensor of inputs, tensor of targets)
    
    def __len__(self):
        return self.size

print('Dataset OK')

from collections import namedtuple
from torchvision import models

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class PerceptualLoss(nn.Module):
    def __init__(self, vgg):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg  # Pretrained VGG model
        self.criterion = nn.MSELoss()  # You can also use L2 loss (MSE)
        # Layer weights can emphasize certain layers more (optional)
        self.layer_weights = {
            'relu1_2': 1.0,
            'relu2_2': 1.0,
            'relu3_3': 1.0,
            'relu4_3': 1.0
        }

    def forward(self, generated, target):
        # Extract VGG features for both images
        generated_features = self.vgg(generated)
        target_features = self.vgg(target)
        
        loss = 0
        # Compute perceptual loss across selected layers
        for layer in self.layer_weights:
            loss += self.layer_weights[layer] * self.criterion(
                getattr(generated_features, layer), # generated_features is an instance of VggOutputs class
                getattr(target_features, layer)
            )
        '''
        getattr(object, attr_name) is a Python built-in function
        that returns the attribute value of an object, 
        where the attribute's name is given as a string.
        since attr_name is a variable not an attribute, you cannot directly use object.attr_name
        '''
        
        return loss
# Initialize VGG-16 feature extractor (requires_grad=False for fixed weights)
vgg_model = Vgg16(requires_grad=False).to(device)

# Initialize Perceptual Loss
perceptual_loss_fn = PerceptualLoss(vgg_model).to(device)


# adding tranformations:
# these probabilistic transforms are not used at the present stage, althout they are retained for my reference and illustration purposes
# deterministic transform is applied in the SRdatasets class itself
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(p = 0.5),
     transforms.RandomVerticalFlip(p = 0.5),
     transforms.RandomRotation(degrees = 90)])

dataset = SRdatasets(dataset_path = 'C:/Users/athar/MLprojects/dataloader_task/Datasets')

train_size = 512*2 # 512*2 = 1024, *2 becuase of deterministic transform (horizontal flip)
# val_size = 17*2
# test_size = 17*2
test_size = 17*4

num_epochs = 5
learning_rate = 1e-3
batch_size = 16


# defining psnr metric:
class PSNR():
    def single(self, pred, target):
        pred = pred.to(device)
        target = target.to(device)
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
            mse_acc = mse_acc + torch.mean((target[i] - predictions[i]) ** 2)/m
            
        if mse_acc == 0:
            return 100
        else:
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse_acc))
        '''
        # more accurate (averaging psnr instead of mse) but more computationally expensive:
        x = 1
        for i in range(predictions.shape[0]):
            mse = torch.mean((target[i] - predictions[i]) ** 2)
            if mse == 0:
                addition = 100/predictions.shape[0]
            else:
                x = x * (PIXEL_MAX/mse)
        return addition + (20 * math.log10(x)) / predictions.shape[0])
        '''

    
    
# defining model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 64,
            kernel_size = 9,
            padding = 4
        )
        self.conv2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 32,
            kernel_size = 1,
            padding = 0
        )
        self.conv3 = nn.Conv2d(
            in_channels = 32,
            out_channels = 3,
            kernel_size = 5,
            padding = 2
        )
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        x = x / 255.0
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        x = 255.0 * x
        return x

model = SRCNN().to(device)
pixel_loss = nn.MSELoss()
optimizer = torch.optim.Adam(
    [
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': learning_rate * 0.1}
    ],
    lr = learning_rate
) # "low learning rate for last layer gives promising results" ~authors of SRCNN paper

batch_loss_agg = 0 # batch loss aggregator (aggregates the loss accross given batch)
batch_loss_list = []

batch_acc_agg = 0
batch_acc_list = []

epoch_loss = 0
epoch_loss_list = []
epoch_acc_list = []


# variables for validation accuracy
batch_val_agg = 0
batch_val_list = []
epoch_val_list = []

num_of_batches = len(train_loader)
prev_epochs = 0

# transfering previous checkpoint
PATH = "srcnn_phase3.pt"
try:
#{
    checkpoint = torch.load(PATH, weights_only = True)
    print('checkpoint loaded successfully')
    transfer = int(input("transfer previous model? 1/0: "))
    if(transfer == 1):
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_loss_list = checkpoint['epoch_loss_list']
        epoch_acc_list = checkpoint['epoch_acc_list']
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
            torch.manual_seed(5)
            # train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
            train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    #}
    else:
        torch.manual_seed(5)
        # train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
#}
except Exception as e:
    print('Exception occured, running without loading checkpoint')
    print(e)


train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True)

# training
gradient_accumulation_steps = 4 # effectivec batch size = 64
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        perceptual_loss = perceptual_loss_fn(outputs, targets)
        loss = pixel_loss(outputs, targets) + 0.1* perceptual_loss

        loss.backward() # computes and accumulates gradients
        
        if( (i+1) % gradient_accumulation_steps == 0 ):
            optimizer.step() # updates parameters after 4 batches (64 images) are analysed
            optimizer.zero_grad() # resets gradients to 0
            
        
        batch_loss_agg = batch_loss_agg + loss.item()/num_of_batches
        # batch_loss_list.append(loss.item())

        step_train_accuracy = PSNR().batch(outputs, targets)
        psnr_input = PSNR().batch(inputs, targets)
        # batch_acc_list.append(step_train_accuracy)
        batch_acc_agg = batch_acc_agg + step_train_accuracy/num_of_batches

        
        if (i + 1) % 16 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, PSNR: {step_train_accuracy:.4f}, diff(pred - input): {(step_train_accuracy - psnr_input):.4f} ')

        # for freeing up gpu space
        del inputs, targets, outputs, loss
        torch.cuda.empty_cache()
        
    epoch_acc_list.append(batch_acc_agg)
    epoch_loss_list.append(batch_loss_agg)
    
    batch_loss_agg = 0
    batch_acc_agg = 0


plt.plot(range(1, prev_epochs + num_epochs+1), epoch_loss_list)
plt.title('loss vs epochs')
plt.show()

plt.plot(range(1, prev_epochs + num_epochs+1), epoch_acc_list)
plt.title('PSNR vs epochs')
print('Training finished')
plt.show()


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

test_acc, predictions = predict(test_loader)
print(f'Test accuracy: {test_acc}')

# displaying some randomly picked examples
print('Some examples: ')
x = 0 # iterations counter
for i, (inputs, targets) in enumerate(test_loader):
    for j in np.random.randint(0, batch_size, 10):
        j = min(len(inputs)-1, j)
        input_arr, target = inputs[j], targets[j]
        input_arr= np.array(input_arr)
        target = np.array(target)
        input_tensor = torch.from_numpy(input_arr).to(device)
        pred = model(input_tensor).detach().cpu().numpy()
        if(len(pred.shape) > 3):
            pred = np.squeeze(pred, 0)
        pred = pred.transpose(1, 2, 0)

        # use of prediction list - didn't work out and felt unnecessary:
        '''
        prediction = np.array(predictions[i, j])
        prediction = prediction.transpose(1, 2, 0)
        '''
        input_arr = input_arr.transpose(1, 2, 0)
        target = target.transpose(1, 2, 0)
        
    
        plt.axis('off')
        
        ax = plt.subplot(1, 3, 1)
        ax.imshow(input_arr.astype('uint8'))
        ax.set_title('input')
        plt.text(0, -100, f'input PSNR: {PSNR().single(torch.from_numpy(input_arr), torch.from_numpy(target)):.4f}')
        ax.axis('off')
    
        ax = plt.subplot(1, 3, 2)
        ax.imshow(pred.astype('uint8'))
        ax.set_title('prediction')
        plt.text(0, -100, f'prediction PSNR: {PSNR().single(torch.from_numpy(pred), torch.from_numpy(target)):.4f}')
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


PATH = "srcnn_phase3.pt"
torch.save({
            'epoch_loss_list': epoch_loss_list,
            'epoch_acc_list': epoch_acc_list,
            'epoch': prev_epochs + num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'train_set': train_set,
            # 'test_set': test_set,
            }, PATH)

print(f"model saved as '{PATH}'")
# stores at: C:\Users\athar\AppData\Roaming\Python\Python312\site-packages
  
