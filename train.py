#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports here
import numpy as np
import pandas as pd

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import vgg16, vgg13
from torch.nn import Sequential, Linear, ReLU, LogSoftmax, NLLLoss, Dropout
from torch.optim import Adam
from collections import OrderedDict
import torch

from time import time
from os.path import isdir
from os import listdir
import json

import argparse
import os
import tempfile
import shutil
import atexit

# In[2]:


class readable_dir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
#             raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
            print(">> Error readable_dir: \'{0}\' is not a valid path".format(prospective_dir))
            exit()
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
#             raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))
            print(">> Error readable_dir: {0} is not a readable dir".format(prospective_dir))
            exit()


# In[3]:


parser = argparse.ArgumentParser(description='This script is used to train a nural network model')
parser.add_argument('data_dir', type=str, action=readable_dir, nargs='?', const= '/', default=None,
                    help='Directory of the data')
parser.add_argument('--save_dir', type=str, action=readable_dir, help='Directory to be saved')

parser.add_argument('-ar','--arch', type=str, help='Architecture of the model', 
                    default = 'vgg16', choices=['vgg16', 'vgg13'])

parser.add_argument('-lr','--learning_rate', type=float, default = 0.001)
parser.add_argument('-hu','--hidden_units', type=int, default = 512)
parser.add_argument('-e','--epochs', type=int, default = 5)

parser.add_argument('-g','--gpu',action='store_true',help='train on the GPU if available');

args = parser.parse_args()

# In[16]:


# args = parser.parse_args('flowers/ -ar vgg13 -e 1 -lr 1e-2 -g --save_dir / '.split())


# In[17]:


# run_args(args)


# In[6]:


def run_args(args):
    if args.gpu and not(torch.cuda.is_available()):
        print('>> Error: GPU is not available in your device')
#         exit()
        return
    else:
        
        device = torch.device('cuda' if args.gpu else 'cpu')
        
        
    data_dir = args.data_dir
    print('>> Loading data from',data_dir)
    
    dataloaders, class_to_idx = loadData(data_dir)
    num_classes = len(class_to_idx)
    print('>> Data loaded successfully')
    print('>> Constructing {} model, with {} output units, and {} hidden units'.format(args.arch,
                                                                                      num_classes, args.hidden_units))
    model = getModel(num_classes,args.hidden_units,
                     arch=args.arch, learning_rate =args.learning_rate, class_to_idx=class_to_idx)
    print('>> Traing model with {} epochs, using the {}'.format(args.epochs, device.type))
    model = trainModel(dataloaders, model, epochs = args.epochs,device = device)
    print('>> Model Trained')
    if args.save_dir:
        print('>> Saving model in directory',args.save_dir)
        save_model(args.save_dir,model)
        print('>> Model saved in {}model_checkpoint.pth'.format(args.save_dir))
        
    
    


# In[7]:


def loadData(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                     ])
    #Transformation for train data set
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])
                                         ])
    #Loading the Images in in ImageFolders
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=train_transform),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms)
                     }

    #Wraping the image folders in dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'],batch_size=32, shuffle=True),
        'valid': DataLoader(image_datasets['valid'],batch_size=32, shuffle=True),
        'test': DataLoader(image_datasets['test'],batch_size=32, shuffle=True)
                     }
    return dataloaders, image_datasets['train'].class_to_idx


# In[8]:


def getModel(output_units,hidden_units,arch='vgg16',class_to_idx=None,
             learning_rate =1e-3, criterion = NLLLoss,optimizer = Adam):
    #Choosing between the models
    if arch == 'vgg13':
        model = vgg13(pretrained=True)
    else:
        model = vgg16(pretrained=True)
    
    #Turning off all the pretrained parameters, since we don't want to retrain them
    for param in model.parameters():
        param.requires_grad = False
    
    #We define our new classifier, with input that match the default vgg16 classifier,
    #and output equals to the number of classes
    classifier = Sequential(OrderedDict([('fc1',Linear(25088,hidden_units)),
                                         ('act1',ReLU()),
                                         ('Dropout1',Dropout(p=0.5)),
                                         ('fc2',Linear(hidden_units,hidden_units)),
                                         ('act2',ReLU()),
                                         ('Dropout2',Dropout(p=0.5)),
                                         ('fc3',Linear(hidden_units,output_units)),
                                         ('output',LogSoftmax(dim=1)),
                                        ]))
    #we deattach the default vgg16 classeifer and plug the one 
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    
    criterion= criterion()
    optimizer = optimizer(model.classifier.parameters(),lr=learning_rate)
    model.criterion = criterion
    model.optimizer = optimizer
    model.arch = arch
    
    return model



# In[10]:


def trainModel(dataloaders, model, epochs = 5, learning_rate =None,
               print_every = 20, device = torch.device('cpu')):
    
    optimizer = model.optimizer
    criterion = model.criterion
    
    if learning_rate:
        b = optimizer.state_dict()
        b['param_groups'][0]['lr'] = learning_rate
        optimizer.load_state_dict(b)
    
    model.to(device);
    
    steps = 0
    running_loss = 0
    start = time()
    for epoch in range(epochs):
        epoch_start = time()
        #We iterate with the training dataloader
        for images, labels in dataloaders['train']:
            steps +=1
            #We move the data to the available device
            images, labels = images.to(device), labels.to(device)

            #clearing the optimizer grads from previous iterations
            optimizer.zero_grad()

            #predcition
            logps = model(images)
            #calcuating loss
            loss = criterion(logps,labels)
            #calcuate backpropgation 
            loss.backward()
            #Take the backpropgation step 
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                valid_loss = 0
                #We measure the running performance of our model with the validition dataset
                with torch.no_grad():
                    for images, labels in dataloaders['valid']:
                        images, labels = images.to(device), labels.to(device)

                        #predcition
                        logps = model(images)
                        #calcuating loss
                        loss = criterion(logps,labels)


                        valid_loss += criterion(logps, labels).item()

                        ps = torch.exp(logps)
                        # Calculate the hardmax, then compare it with the actual classes
                        equality = (labels.data == ps.max(1)[1])
                        # Calcuate the accuracy by taking the mean of the accurate predictions
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                print(
                        "Epoch({}/{})-".format(epoch+1, epochs),
                        "Epoch time: s{:.2f},".format(time()-epoch_start),
                        "Start time: s{:.2f},".format(time()-start),
                        "Training Loss: {:.2f},".format(running_loss/print_every),
                        "Valid Loss: {:.2f},".format(valid_loss/len(dataloaders['valid'])),
                        "Accuracy: %{:.2f}".format(100*accuracy/len(dataloaders['valid']))
                )

                running_loss = 0
                model.train()
    return model


# In[11]:


def save_model(save_dir,model):
    model_checkpoint = {
                    'state_dict': model.state_dict(),
                    'classifier': model.classifier,
                    'class_to_idx': model.class_to_idx,
                    'criterion' : model.criterion,
                    'optimizer' : model.optimizer,
                    'arch': model.arch
                    }
    if save_dir =='/':
        save_dir = ''
    torch.save(model_checkpoint, save_dir+'model_checkpoint_p2_save.pth');
    
if __name__ == '__main__':
    run_args(args)

