import DataMaker as dm
import ModelTrainer as mt
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim 


TRAIN_FILE = 'datasets/en_atis-ud-train.conllu'
DEV_FILE = 'datasets/en_atis-ud-dev.conllu'
TEST_FILE = 'datasets/en_atis-ud-test.conllu'
EMBEDDING_DIM = int(input('Enter the embedding dimension : '))
HIDDEN_DIM = int(input('Enter the hidden dimension : '))
LEARNING_RATE = float(input('Enter the learning rate : '))
print()
jugaad = (input('Do you want to train the model with jugaad ? (y/n) : ') == 'y')
EPOCHS = int(input('Enter the number of epochs : '))
BATCH_SIZE = 32 

DEVICE = mt.get_device()
print('Device : ',DEVICE)
print('Loading data...')
WORD2IDX = { '<PAD>':0,'<UNK>':1}
TAG2IDX = {'<PAD>':0 , '<UNK>':1} 

CRITERION = nn.NLLLoss()
Loaders={}
Loaders['train'] = dm.Get_Dataloader(file_path= TRAIN_FILE , word2idx=WORD2IDX , tag2idx=TAG2IDX , batch_size=BATCH_SIZE , train=True, jugaad=jugaad)
Loaders['dev'] = dm.Get_Dataloader(file_path= DEV_FILE , word2idx=WORD2IDX , tag2idx=TAG2IDX , batch_size=BATCH_SIZE , jugaad=jugaad)
Loaders['test'] = dm.Get_Dataloader(file_path= TEST_FILE , word2idx=WORD2IDX , tag2idx=TAG2IDX , batch_size=BATCH_SIZE, jugaad = False)

print('Data loaded')
print('Training model...')
model = mt.make_model(embedding_dim=EMBEDDING_DIM , hidden_dim=HIDDEN_DIM , word2idx=WORD2IDX , tag2idx=TAG2IDX)
OPTIMIZER = optim.Adam(model.parameters() , lr=LEARNING_RATE)
model = model.to(DEVICE)
model = mt.train_model(model , train_loader= Loaders['train'] ,dev_loader= Loaders['dev'] , optimizer=OPTIMIZER , criterion=CRITERION , epochs=EPOCHS , device=DEVICE ,path='model1.pt', save=True
        ,word2idx=WORD2IDX , tag2idx=TAG2IDX)
print('Model trained')
print('Testing model...')
mt.test_model(model , test_loader=Loaders['test'] , device=DEVICE)
print('Model tested')
print(mt.get_classification(model , Loaders['test'] , device=DEVICE , word2idx=WORD2IDX , tag2idx=TAG2IDX))