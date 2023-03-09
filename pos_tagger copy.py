#importing required libraries 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim 
from conllu import parse
import numpy as np
from torch.utils.data import Dataset, DataLoader

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
    


def get_device():
    '''
    This function returns the device on which the model will be trained
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


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

def create_datasets(file_path , word2idx , tag2idx , train=False):
    '''
    This function creates the dataset from the raw data
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
                    tag2idx.update({word['upostag']:len(tag2idx)})
                sentence_tags.append(tag2idx[word['upostag']])
                sentence_words.append(word2idx[word['form']])
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
                    tag2idx.update({new_tag:len(tag2idx)})
                sentence_tags.append(tag2idx[new_tag])
            dataset[1].append(sentence_tags)
            dataset[0].append(sentence_words)
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



class POS_tagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(POS_tagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
       
    def forward(self, batch):
        embeds = self.word_embeddings(batch)
        lstm_out, _ = self.lstm(embeds.permute(1, 0, 2))
        # print(lstm_out.shape)
        # print('lstm_out')
        # print(lstm_out[-1].shape)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)
        # print(tag_scores.shape)
        return tag_scores.permute(1, 2, 0)


def make_model(embedding_dim , hidden_dim , word2idx , tag2idx ):
    model = POS_tagger(embedding_dim , hidden_dim , len(word2idx) , len(tag2idx))
    return model


def train_model(model , train_loader , dev_loader, optimizer , criterion , epochs , device, path):
    ''' 
    This function trains the model
    args : model , train_loader , optimizer , criterion , epochs , device
    returns : trained model
    '''
    model.train()
    steps = 0
    running_loss = 0
    print_every = 100
    best_accuracy = 0
    for epoch in range(epochs):
        for words , tags in train_loader:
            # print(words.shape , tags.shape)
            steps += 1
            words = words.to(device)
            tags = tags.to(device)
            optimizer.zero_grad()
            
            output = model(words)

            # print(output.shape, tags.shape)
            # print('output')
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
               
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict() , path)
                print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Test accuracy: {accuracy/len(dev_loader):.3f}")
                running_loss = 0
                model.train()
            
    return model


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


def collate_jugaad(dataset):
    '''
    This function is used to pad the sentences in the batch
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