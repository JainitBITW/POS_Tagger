#author : Jainit Bafna 
#date : 9/3/2023
#description : This file contains the code to create the dataset for the model


#importing required libraries 
import torch 

from conllu import parse

from torch.utils.data import Dataset, DataLoader
import random 


# class to create the dataset
class POS_Dataset(Dataset):
    '''
    This class creates the dataset for the model
    '''

    def __init__(self ,words , tags , word2idx ):
        '''
        This function initializes the dataset
        words contains the list of sentences in word form
        args : data_dir , word2idx , tag2idx
        
        '''
        self.words = words.clone().detach()
        self.tags = tags.clone().detach()
        self.word2idx = word2idx

    def __len__(self):
        '''
        This function returns the length of the dataset
        '''
        return len(self.words)
    
    def __getitem__(self , idx):
        '''
        This function returns the item at the given index
        '''
        return self.words[idx] , self.tags[idx]


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

def create_datasets(file_path , word2idx , tag2idx , train=False , jugaad=True):
    '''
    This function creates the dataset from the raw data jugaad is basically a flag to apply unknown learning to the dataset
    args : raw_data
    returns : dataset where each element is a tuple of (sentence , tags)
    '''
    dataset = [[] , []]
    if train: 
        raw_data = read_file(file_path)
        for sentence in raw_data:
            sentence_words = []
            sentence_tags = []

            for word in sentence:
                if word['form'] not in word2idx:
                    word2idx.update({word['form']:len(word2idx)})
                if word['upostag'] not in tag2idx:
                    # print(word['upostag'])
                    tag2idx.update({word['upostag']:len(tag2idx)})
                sentence_tags.append(tag2idx[word['upostag']])
                sentence_words.append(word2idx[word['form']])
            dataset[0].append(sentence_words)
            dataset[1].append(sentence_tags)
            if jugaad:
                if random.random() < 0.6:
                    rand_int = random.randint(0,len(sentence_words)-1)
                    for i in range(rand_int):
                        rand_int2 = random.randint(0,len(sentence_words)-1)
                        sentence_words[rand_int2] = word2idx['<UNK>']
            dataset[0].append(sentence_words)
            dataset[1].append(sentence_tags)
        return collate_jugaad(dataset)
    else:
        raw_data= read_file (file_path)
        for sentence in raw_data:
            sentence_words = []
            sentence_tags = []
            for word in sentence:
                new_word = word['form']
                new_tag = word['upostag']
                if new_word not in word2idx:
                    new_word = '<UNK>'
                sentence_words.append(word2idx[new_word])
                if new_tag not in tag2idx:
                    new_tag = '<UNK>'
                sentence_tags.append(tag2idx[new_tag])
            dataset[1].append(sentence_tags)
            dataset[0].append(sentence_words)
            if jugaad:
                if random.random() < 0.6:
                    rand_int = random.randint(0,len(sentence_words)-1)
                    for i in range(rand_int):
                        rand_int2 = random.randint(0,len(sentence_words)-1)
                        sentence_words[rand_int2] = word2idx['<UNK>']
            dataset[0].append(sentence_words)
            dataset[1].append(sentence_tags)
        return collate_jugaad(dataset)
    

def dataset_loader(dataset , batch_size, word2idx):
    '''
    This function creates the dataloader from the dataset
    args : dataset , batch_size
    returns : dataloader
    '''
    # print(dataset)
    dataset_custom = POS_Dataset(dataset[0] , dataset[1] , word2idx)
    
    data_loader = DataLoader(dataset_custom , batch_size = batch_size , shuffle = True)

    return data_loader

def collate_jugaad(dataset):
    '''
    This function is used to pad the sentences in the batch. This is basically a jugaad instead of using the collate function.
    args : dataset
    returns : padded sentences , tags
    '''
    max_len = max(len(sentence) for sentence in dataset[0])
    # print(max_len)
    sentences = dataset[0]
    tags = dataset[1]
    for i in range(len(sentences)):
        sentences[i] = sentences[i] + [0]*(max_len - len(sentences[i]))
        tags[i] = tags[i] + [0]*(max_len - len(tags[i]))
    return [torch.tensor(sentences) , torch.tensor(tags)]


def Get_Dataloader(file_path , word2idx , tag2idx , batch_size , train=False, jugaad = True):
    '''
    This function creates the dataloader from the file
    args : file_path , word2idx , tag2idx , batch_size
    returns : dataloader
    '''
    dataset = create_datasets(file_path , word2idx , tag2idx , train , jugaad= jugaad)
    dataloader = dataset_loader(dataset , batch_size , word2idx)
    return dataloader

if __name__ == '__main__':
    print("This is the DataMaker.py file")

