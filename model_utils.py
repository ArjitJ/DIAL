import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


config = RobertaConfig(vocab_size=50265, max_position_embeddings=514, num_attention_heads=12, num_hidden_layers=6, type_vocab_size=1)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_len=512)
MASK_PROB = 0.5


def sim(x, y):
    return -((x-y)**2).sum(dim=-1)

def criterion(u, v, rand_u, rand_v):
    pos = sim(u,v)
    return pos.mean() - torch.log(pos.exp().sum() + sim(u,rand_v).exp().sum() + sim(rand_u,v).exp().sum() + sim(rand_u,rand_v).exp().sum())


class RobertaClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.bitmask = (torch.rand((1, 768)) < MASK_PROB).float().cuda()
        self.bitmask.requires_grad = False
        self.dense = nn.Linear(768, 768)

    def forward(self, features, **kwargs):
        x = features*self.bitmask
        x = self.dense(x)
        x = torch.tanh(x)
        return x

class PairedRobertaClassificationHead(nn.Module):
    def __init__(self, numlabel=2):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768, numlabel)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Model(nn.Module):
    def __init__(self, numEmbeddings):
        super(Model, self).__init__()
        self.numEmbeddings = numEmbeddings
        self.transformer = RobertaModel(config=config).from_pretrained('roberta-base', config=config)
        self.transformer.train()
        self.fc = nn.ModuleList([RobertaClassificationHead() for i in range(numEmbeddings)])
        self.fc_paired = PairedRobertaClassificationHead()
        for param in self.transformer.parameters():
            param.requires_grad = True
        
    def forward_unpaired(self, x, y):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            attn_x = (x != tokenizer.pad_token_id).float().cuda()
            attn_y = (y != tokenizer.pad_token_id).float().cuda()
            x = self.transformer(input_ids = x, attention_mask=attn_x)[0]
            y = self.transformer(input_ids = y, attention_mask=attn_y)[0]
            x = (x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
            y = (y*attn_y.unsqueeze(-1)).sum(1)/attn_y.unsqueeze(-1).sum(1)
        embeddings_x = []
        embeddings_y = []
        for i in range(self.numEmbeddings):
            u, v = self.fc[i](x), self.fc[i](y)
            embeddings_x.append(u)
            embeddings_y.append(v)
        return embeddings_x, embeddings_y
    
    def forward_paired(self, x):
        x = x.cuda()
        attn_x = (x != tokenizer.pad_token_id).float().cuda()
        x = self.transformer(input_ids = x, attention_mask=attn_x)[0]
        x = (x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
        return self.fc_paired(x)