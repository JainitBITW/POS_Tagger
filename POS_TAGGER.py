import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from conllu import parse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import ModelTrainer as mt 

FILE='model1.pt'
checkpoint = torch.load(FILE)
WORD2IDX = checkpoint['word2idx']
TAG2IDX = checkpoint['tag2idx']
EMBEDDING_DIM = checkpoint['embedding_dim']
HIDDEN_DIM = checkpoint['hidden_dim']
model = mt.make_model(embedding_dim=EMBEDDING_DIM , hidden_dim=HIDDEN_DIM , word2idx=WORD2IDX , tag2idx=TAG2IDX)
model.load_state_dict(checkpoint['model_state_dict'])

input_sentence = input('Enter the sentence : ')
print('\n')
to_input = input_sentence.lower().split()
input_sentence=input_sentence.split()
to_input = [WORD2IDX[word] if word in WORD2IDX else WORD2IDX['<UNK>'] for word in to_input]
to_input = torch.tensor(to_input).reshape(1,-1)
ps = model(to_input)
ps = torch.exp(ps)
top_p , top_class = ps.topk(1 , dim=1)
top_class= top_class[0][0].tolist()
idx2tag = {v:k for k,v in TAG2IDX.items()}
for i in range(len(input_sentence)):
    print(f'{input_sentence[i]} \t {idx2tag[top_class[i]]}')
