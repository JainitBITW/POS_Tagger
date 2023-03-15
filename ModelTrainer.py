#author : Jainit Bafna
#date : 9/3/2023
#description : This file contains the code to train a model 

#importing the necessary libraries
from sklearn.metrics import classification_report
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim 
from conllu import parse
import numpy as np
from torch.utils.data import Dataset, DataLoader


def get_device():
    '''
    This function returns the device on which the model will be trained
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


class POS_tagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(POS_tagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)  # output size doubled due to bidirectional LSTM

    def forward(self, batch):
        embeds = self.word_embeddings(batch)
        lstm_out, _ = self.lstm(embeds.permute(1, 0, 2))
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores.permute(1, 2, 0)


def make_model(embedding_dim , hidden_dim , word2idx , tag2idx ):
    model = POS_tagger(embedding_dim , hidden_dim , len(word2idx) , len(tag2idx))
    return model




def make_model(embedding_dim , hidden_dim , word2idx , tag2idx ):
    model = POS_tagger(embedding_dim , hidden_dim , len(word2idx) , len(tag2idx))
    return model


def train_model(model , train_loader , dev_loader, optimizer , criterion , epochs , device, path , save=True, word2idx=None, tag2idx=None):
    ''' 
    This function trains the model
    args : model , train_loader , optimizer , criterion , epochs , device
    returns : trained model
    '''
    model.train()
    model = model.to(device)
    print(model.hidden_dim)
    print(model.embedding_dim)
    steps = 0
    running_loss = 0
    print_every = 100
    best_accuracy = 0
    print('Printing only when accuracy increases')
    for epoch in range(epochs):
        for words , tags in train_loader:
            steps += 1
            words = words.to(device)
            tags = tags.to(device)
            optimizer.zero_grad()
            output = model(words)
            loss = criterion(output , tags)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                with torch.no_grad():
                    for words , tags in dev_loader:
                        words = words.to(device)
                        tags = tags.to(device)
                        output = model(words)
                        top_p , top_class = output.topk(1 , dim=1)
                        equals = top_class == tags.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                if accuracy > best_accuracy and save:
                    best_accuracy = accuracy
                    torch.save({
                       'word2idx' : word2idx,
                          'tag2idx' : tag2idx,
                            'model_state_dict' : model.state_dict(),
                            'optimizer_state_dict' : optimizer.state_dict(),
                            'hidden_dim' : model.hidden_dim,
                            'embedding_dim' : model.embedding_dim,
    
                    }, path)
                    print(f"Epoch {epoch+1}/{epochs}.. "
                            f"Train loss: {running_loss/print_every:.3f}.. "
                            f"Test accuracy: {accuracy/len(dev_loader):.3f}")
                running_loss = 0
                model.train()
            
    return model


def test_model(model , test_loader , device):
    '''
    This function tests the model
    args : model , test_loader , device
    returns : accuracy of the model
    '''
    model = model.to(device)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for words , tags in test_loader:
            words = words.to(device)
            tags = tags.to(device)
            output = model(words)
            top_p , top_class = output.topk(1 , dim=1)
            # print(top_class.shape)
            equals = top_class == tags.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    print(f"Test accuracy : {accuracy/len(test_loader)}")
    return accuracy/len(test_loader)

def get_classification(model , test_loader , device , word2idx , tag2idx):
    '''
    This function returns the classification report of the model
    args : model , test_loader , device , word2idx , tag2idx
    returns : classification report
    '''
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for words , tags in test_loader:
            words = words.to(device)
            tags = tags.to(device)
            output = model(words)
            top_p , top_class = output.topk(1 , dim=1)
            y_true.extend(tags.view(-1).cpu().numpy())
            y_pred.extend(top_class.view(-1).cpu().numpy())
    tagss= set(y_true)
    idx2tag = {v:k for k,v in tag2idx.items()}
    target_names = [idx2tag[i] for i in tagss]
    print(classification_report(y_true , y_pred , target_names=target_names))
    return classification_report(y_true , y_pred)


