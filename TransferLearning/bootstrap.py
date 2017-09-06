#!/usr/bin/env python
# Script for downloading the Oxford17 flower dataset and the Oxford102 Caffe model

"""
The following is modified from: https://github.com/jimgoo/caffe-oxford102/blob/master/bootstrap.py

Note that there are no labels associated with the Oxford17 flower dataset and that here we only download
the Oxford102 model that is based on AlexNet, not VGG-S
"""

import os
import sys
import glob
import urllib
import tarfile
import numpy as np
from scipy.io import loadmat

# Function for downloading files
def download_file(url, dest=None):
    """Returns the trn, val, or tst split of the data from datasplits.mat
    Args:
      url (string): where the data is located on the web
      dest (string): where to save the data
    Returns:
      Downloads the data into the corresponding dest location
    """
    if not dest:
        dest = 'data/' + url.split('/')[-1]
    else: # added for creating separate directories other than data
        dest += '/' + url.split('/')[-1]
    urllib.urlretrieve(url, dest)
    
# Download the Oxford102 dataset into the current directory
if not os.path.exists('data'):
    os.mkdir('data')
    
    print("Downloading images...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz')
    tarfile.open("data/17flowers.tgz").extractall(path='data/')
    
    # The main download page for the dataset doesn't come with image labels, but we can deduce them easily
    # There are 17 flower classes, with 80 examples each- these are already ordered to begin with
    #print("Downloading image labels...")
    #download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat')
    
    print("Downloading train/test/valid splits...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat')

if not os.path.exists('caffe/oxford102.caffemodel'):
    os.mkdir('caffe')
    print("Downloading Oxford102 pretrained weights...")
    download_file('https://s3.amazonaws.com/jgoode/oxford102.caffemodel','caffe')
    
if not os.path.exists('caffe/deploy.prototxt'):
    print("Downloading Oxford102 network structure...")
    download_file('https://raw.githubusercontent.com/jimgoo/caffe-oxford102/master/AlexNet/deploy.prototxt','caffe')
    

# We'll just use the AlexNet version of Oxford-102, so we don't need the VGG_S model
"""
if not os.path.exists('VGG_S/pretrained-weights.caffemodel'):
    print('Downloading VGG_S pretrained weights...')
    download_file('http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_S.caffemodel',
                  'VGG_S/pretrained-weights.caffemodel')
"""    
    
# Clean up by removing the compressed data directory
os.remove('data/17flowers.tgz')