import DataMaker as dm
import ModelTrainer as mt
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim 


TRAIN_FILE = 'en_atis-ud-train.conllu'
DEV_FILE = 'en_atis-ud-dev.conllu'
TEST_FILE = 'en_atis-ud-test.conllu'
EMBEDDING_DIM = 100 
HIDDEN_DIM = 100 
EPOCHS = 10
BATCH_SIZE = 32 
LEARNING_RATE = 0.001
DEVICE = mt.get_device()
WORD2IDX = { '<PAD>':0,'<UNK>':1}
TAG2IDX = {'<PAD>':0} 

CRITERION = nn.NLLLoss()
Loaders={}
Loaders['train'] = dm.Get_Dataloader(file_path= TRAIN_FILE , word2idx=WORD2IDX , tag2idx=TAG2IDX , batch_size=BATCH_SIZE , train=True, jugaad=True)
Loaders['dev'] = dm.Get_Dataloader(file_path= DEV_FILE , word2idx=WORD2IDX , tag2idx=TAG2IDX , batch_size=BATCH_SIZE , jugaad=True)
Loaders['test'] = dm.Get_Dataloader(file_path= TEST_FILE , word2idx=WORD2IDX , tag2idx=TAG2IDX , batch_size=BATCH_SIZE, jugaad = False)

model = mt.make_model(embedding_dim=EMBEDDING_DIM , hidden_dim=HIDDEN_DIM , word2idx=WORD2IDX , tag2idx=TAG2IDX)
OPTIMIZER = optim.Adam(model.parameters() , lr=LEARNING_RATE)
model = model.to(DEVICE)
model = mt.train_model(model , train_loader= Loaders['train'] ,dev_loader= Loaders['dev'] , optimizer=OPTIMIZER , criterion=CRITERION , epochs=EPOCHS , device=DEVICE ,path='model1.pt', save=True
        ,word2idx=WORD2IDX , tag2idx=TAG2IDX)
mt.test_model(model , test_loader=Loaders['test'] , device=DEVICE)
