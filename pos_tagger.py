#importing required libraries 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim 
from conllu import parse
import numpy as np


#reading data from the file 
def read_file(file_name):
    '''
    This function reads the file and returns the data in the form of list of sentences.
    files shoulf be in conllu format 
    conllu file is parsed using conllu library
    args : file_name
    returns : list of sentences
    '''
    