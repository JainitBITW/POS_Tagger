#importing required libraries 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim 
from conllu import parse
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SentenceDataset(Dataset):
    



#reading data from the file 
def read_file(file_name):
    '''
    This function reads the file and returns the data in the form of list of sentences.
    files shoulf be in conllu format 
    conllu file is parsed using conllu library
    args : file_name
    returns : list of sentences
    '''
    
    raw_data = parse(open(file_name).read())
    return raw_data


def create_dataset(raw_data , word2idx , tag2idx):
    '''
    This function creates the dataset from the raw data
    args : raw_data
    returns : dataset
    '''
    dataset =[]
    for sentence in raw_data : 
        sentence_list =[[],[]]
        for word in sentence:
            if word['form'] in word2idx:
                word2idx[word['form']] = len(word2idx)
            if word['upos'] in tag2idx:
                tag2idx[word['upos']] = len(tag2idx)
            sentence_list[0].append(word2idx[word['form']])
            sentence_list[1].append(tag2idx[word['upos']])
        dataset.append(sentence_list)
    return dataset





def dataset_loader(dataset , batch_size):
    '''
    This function creates the dataloader from the dataset
    args : dataset , batch_size
    returns : dataloader
    '''
    data = torch.utils.data.DataLoader(dataset , batch_size = batch_size , shuffle = True)

    return data