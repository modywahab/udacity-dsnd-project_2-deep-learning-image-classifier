#!/usr/bin/env python
# coding: utf-8

# In[80]:


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


from PIL import Image
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
            print("readable_dir:{0} is not a valid path".format(prospective_dir))
            exit()
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            print("readable_dir:{0} is not a readable dir".format(prospective_dir))
            exit()

class readable_file(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isfile(prospective_dir):
            print("readable_dir:{0} is not a valid path".format(prospective_dir))
            exit()
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            print("readable_dir:{0} is not a readable dir".format(prospective_dir))
            exit()

# In[3]:


parser = argparse.ArgumentParser(
    description='This script is used to give prediction of an input on a checkpointed model')
parser.add_argument('input', type=str, action=readable_file, help='Directory of the input image')
parser.add_argument('checkpoint', type=str, action=readable_file, help='Directory of the checkpoint')

parser.add_argument('-cn','--category_names', type=str, action=readable_file, help='Directory json file')
parser.add_argument('-tk','--top_k', type=int, default = 5)
parser.add_argument('-g','--gpu',action='store_true',help='train on the GPU if available');
args = parser.parse_args()


# In[139]:


def run_args(args):
    if args.gpu and not(torch.cuda.is_available()):
        print('Error: GPU is not available in your device')
#         exit()
        return
    else:
        device = torch.device('cuda' if args.gpu else 'cpu')
    
    print('>> Loading the model from the checkpoint')
    model, model_checkpoint = load_model(args.checkpoint,device=device)
    print('>>> Model loaded')
    print('>> Predicting the class of the input, using',device.type)
    probs, labels = predict(args.input, model, topk=args.top_k, device=device)
    true_cat = args.input.split('/')[-2]
    #Getting the names of the predicted classes
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        true_cat = cat_to_name[true_cat]
        labels = [cat_to_name[index] for index in labels]
    
    print('>>> The True class is \'{}\', and the models predictions are:'.format(true_cat))
    for i in range(len(labels)):
        prefix = '\t {}-'.format(i+1)
        print(progressBar(probs[i], sum(probs), prefix = prefix, suffix = labels[i], length = 50))
    print('CAUTION: These are not the real probabilities, but the precentages compared with the probability between the top',args.top_k)
        
        
    


# In[143]:


#Args example
# 'flowers/test/13/image_05767.jpg load_check_point.pth -cn cat_to_name.json -tk 5 -g'


# In[71]:


def load_model(checkpoint,device=torch.device('cpu')):
    #Load the checkpoint from the same path
    #map_l
    if device.type == 'cuda':
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    model_checkpoint_load = torch.load(checkpoint,map_location=map_location)
    #reconstruct the model
    if model_checkpoint_load['arch'] == 'vgg13':
        arch = vgg13
    else:
        arch = vgg16
        
    model_load = arch(pretrained=True)
    #Turn off the parameters
    for param in model_load.parameters():
        param.requires_grad = False
    #Match the stored variables from the checkpoint
    model_load.classifier = model_checkpoint_load['classifier']
    model_load.class_to_idx = model_checkpoint_load['class_to_idx']
    model_load.load_state_dict(model_checkpoint_load['state_dict'])
    
    return model_load, model_checkpoint_load


# In[10]:


def process_image(path,norm=True,to_np=True,tensor=False):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(path)
    width, height = image.size
    #Resizing the images where the shortest side is 256 pixels, keeping the aspect ratio
    min_dim = 256
    argmin_dim = np.argmin(image.size)
    argmax_dim = np.abs(argmin_dim-1)
    max_dim = int(image.size[argmax_dim]*(min_dim/float(image.size[argmin_dim])))
    image_size = [0,0]
    image_size[argmin_dim] = min_dim
    image_size[argmax_dim] = max_dim
    image = image.resize(tuple(image_size), Image.ANTIALIAS)
    
#     if width < height: resize_size=[256, 256**600]
#     else: resize_size=[256**600, 256]
        
#     image.thumbnail(size=resize_size)
    
    ##Center crop to 224*224
    c_x,c_y = width/4, height/4
    image = image.crop((c_x-(244/2), c_y-(244/2), c_x+(244/2), c_y+(244/2)))
    
    
    
    image = np.array(image)/255
    #Normalize the pixles to match the trained data
    if norm:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = ( image  - mean)/std
    image = image.transpose(2, 0, 1)
    
    if to_np==False:
        image = image.transpose((1, 2, 0))
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    elif tensor==True:
        #if ture the function will return a tensor
        image = np.expand_dims(image,axis=0)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        
        
    return image


# In[82]:


def predict(image_path, model, topk=5,device=torch.device('cpu')):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #Switch to the availble device
    model.to(device)
    
    #Make the predection
    logs = model(process_image(image_path,tensor=True).to(device))
    probs = logs.exp()
    
    #Get the top-k probabilities
    topk = probs.topk(topk)
    top_probs = topk[0][0].tolist()
    
    #Switch to the orginal class index
    top_cat = [int(cat) for cat in topk[1][0].tolist()]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[index] for index in top_cat]
    return top_probs, classes


# In[142]:
def progressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    return str('%s |%s| %s%% %s' % (prefix, bar, percent, suffix))

if __name__ == '__main__':
    run_args(args)

