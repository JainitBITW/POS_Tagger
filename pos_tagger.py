#importing required libraries 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim 
from conllu import parse
import numpy as np
from torch.utils.data import Dataset, DataLoader




def test_model(model , test_loader , device):
    model = model.to(device)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for words , tags in test_loader:
            words = words.to(device)
            tags = tags.to(device)
            output = model(words)
            top_p , top_class = output.topk(1 , dim=1)
            equals = top_class == tags.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    print(f"Test accuracy : {accuracy/len(test_loader)}")
    return accuracy/len(test_loader)


if __name__ == '__main__':
    #hyperparameters
    embedding_dim = 128 
    hidden_dim = 128
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    device = get_device()
    word2idx = { '<PAD>':0,'<UNK>':1}
    tag2idx = {'<PAD>':0}
    #reading data
    train_data = create_datasets('en_atis-ud-train.conllu' , word2idx , tag2idx , train=True)
    dev_data = create_datasets('en_atis-ud-dev.conllu' , word2idx , tag2idx)
    test_data = create_datasets('en_atis-ud-test.conllu' , word2idx , tag2idx)
    #creating dataloaders
    train_loader = dataset_loader(train_data , batch_size, word2idx)
    dev_loader = dataset_loader(dev_data , batch_size,word2idx)
    test_loader = dataset_loader(test_data , batch_size,word2idx)
 
 
    model = make_model(embedding_dim , hidden_dim , word2idx , tag2idx)
   
    model = model.to(device)
    #creating optimizer and criterion
    optimizer = optim.Adam(model.parameters() , lr=learning_rate)
    criterion = nn.NLLLoss()
    #training model
    model = train_model(model , train_loader , dev_loader , optimizer , criterion , epochs , device , 'model.pt')

    input = 'i am a boy i love cookies'
    input = input.split()
    input_new = []
    for word in input:
        if word in word2idx:
            input_new.append(word2idx[word])
        else:
            input_new.append(word2idx['<UNK>'])
    input_new = torch.tensor(input_new)
    input_new= input_new.reshape(1,-1)
    tags_scores = model(input_new)
    ps = torch.exp(tags_scores)
    top_p , top_class = ps.topk(1 , dim=1)
    top_class = top_class[0][0]
    top_class= top_class.tolist()
    idxtag = {v:k for k,v in tag2idx.items()}
    for i in range(len(input)):
        print(f'{input[i]} \t {idxtag[top_class[i]]}')
    #testing model
    print('Testing model')
    test_model(model , test_loader , device)