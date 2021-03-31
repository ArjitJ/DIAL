import argparse
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from transformers import AdamW, RobertaConfig, RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from dataloader import MyDataset
from itertools import cycle
from statistics import mean

config = RobertaConfig(
    vocab_size=50265,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_len=512)

parser = argparse.ArgumentParser()
parser.add_argument("--numEmbeddings", type=int, default=3)
parser.add_argument("--indices", type=str, default='', required=True)
parser.add_argument("--out", type=str, default='', required=True)
parser.add_argument("--data", type=str, default='', required=True)
parser.add_argument("--norm", action='store_true')
parser.add_argument("--method", type=str, choices=['l2', 'l1', 'cosine'], default='l2')

args = parser.parse_args()
datasetColumns = None
dataDir = None
numSampleRetrieve = None
k=3
LB = 128
if args.data == 'walmart_amazon_exp':
    datasetColumns = ['title', 'category', 'brand', 'modelno', 'price']
    dataDir = 'data/walmart_amazon_exp/'
    numSampleRetrieve = 22074
elif args.data == 'amazon_google_exp':
    datasetColumns = ['title', 'manufacturer', 'price']
    dataDir = 'data/amazon_google_exp/'
    numSampleRetrieve = 3226
elif args.data == 'dblp_acm_exp':
    datasetColumns = ['title', 'authors', 'venue', 'year']
    dataDir = 'data/dblp_acm_exp/'
    numSampleRetrieve = 2294
elif args.data == 'abt_buy_exp':
    datasetColumns = ['name', 'description', 'price']
    dataDir = 'data/abt_buy_exp/'
    numSampleRetrieve = 1091
    k=20
elif args.data == 'dblp_scholar_exp':
    datasetColumns = ['title', 'authors', 'venue', 'year']
    dataDir = 'data/dblp_scholar_exp/'
    numSampleRetrieve = 64263
else:
    exit(1)



torch.manual_seed(0)
np.random.seed(0)
indices = np.loadtxt(dataDir + args.indices, delimiter=',').astype(int)
numEmbeddings = args.numEmbeddings
len_indices = len(indices)//LB

class MyPairedDataset(MyDataset):
    def __init__(self, file_path, indices, tokenizer, datasetColumns):
        super(MyPairedDataset, self).__init__(file_path, indices, tokenizer, datasetColumns)
        self.mode = 'train'
        self.test = np.loadtxt(file_path + 'test.txt', delimiter=',').astype(int)
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.indices)
        else:
            return self.test.shape[0]

    def __getitem__(self, i):
        if self.mode == 'train':
            x = self.indices[i, 0]
            y = self.indices[i, 1]
            lbl = self.labels[i]
        else:
            x = self.test[i, 0]
            y = self.test[i, 1]
            lbl = 1 if [x, y] in self.matches else 0
        xexample = self.xexamples[x][1:-1]
        yexample = self.yexamples[y][1:-1]
            
        input_tokens = []
        input_tokens += [tokenizer.cls_token_id]
        input_tokens += xexample
        input_tokens += [tokenizer.sep_token_id]
        input_tokens += [tokenizer.sep_token_id]
        input_tokens += yexample
        input_tokens += [tokenizer.sep_token_id]
        
        
        return torch.tensor(input_tokens, dtype=torch.long), lbl

class MyPositiveDataset(MyDataset):
    def __init__(self, file_path, indices, tokenizer, datasetColumns):
        super(MyPositiveDataset, self).__init__(file_path, indices, tokenizer, datasetColumns)
        mask = self.labels == 1
        self.indices = self.indices[mask]
        self.labels = self.labels[mask]

class MyNegativeDataset(MyDataset):
    def __init__(self, file_path, indices, tokenizer, datasetColumns):
        super(MyNegativeDataset, self).__init__(file_path, indices, tokenizer, datasetColumns)
        mask = self.labels == 0
        self.indices = self.indices[mask]
        self.labels = self.labels[mask]
        
class MyRandomDataset(MyDataset):
    def __init__(self, file_path, indices, tokenizer, datasetColumns):
        super(MyRandomDataset, self).__init__(file_path, indices, tokenizer, datasetColumns)
        self.indices = self.indices[self.labels == 1]
        
    def __getitem__(self, i):
        x = np.random.randint(len(self.xexamples))
        y = np.random.randint(len(self.yexamples))
        xexample = self.xexamples[x]
        yexample = self.yexamples[y]
        return torch.tensor(xexample, dtype=torch.long),  torch.tensor(yexample, dtype=torch.long),  0*self.labels[i]        
        
        
def _tensorize_batch(examples):
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)
    else:
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({self.tokenizer.__class__.__name__}) does not have one."
            )
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

def collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    target = [item[2] for item in batch]
    target = torch.Tensor(target)
    
    xbatch = _tensorize_batch(x)
    ybatch = _tensorize_batch(y)
    
    return xbatch, ybatch, target

def paired_collate_fn(batch):
    x = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.Tensor(target)
    
    xbatch = _tensorize_batch(x)
    
    return xbatch, target

class RobertaClassificationHead(nn.Module):
    def __init__(self, numlabel=2):
        super().__init__()
        self.bitmask = (torch.rand((1, 768)) < 0.5).float().cuda()
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
            embed_x = self.transformer(input_ids = x, attention_mask=attn_x)[0]
            embed_y = self.transformer(input_ids = y, attention_mask=attn_y)[0]
            embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
            embed_y = (embed_y*attn_y.unsqueeze(-1)).sum(1)/attn_y.unsqueeze(-1).sum(1)
        embeddings_x = []
        embeddings_y = []
        for i in range(self.numEmbeddings):
            u, v = self.fc[i](embed_x), self.fc[i](embed_y)
            embeddings_x.append(u)
            embeddings_y.append(v)
        return embeddings_x, embeddings_y
    
    def forward_paired(self, x):
        x = x.cuda()
        attn_x = (x != tokenizer.pad_token_id).float().cuda()
        embed_x = self.transformer(input_ids = x, attention_mask=attn_x)[0]
        embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
        return self.fc_paired(embed_x)

paireddataset = MyPairedDataset(dataDir, indices, tokenizer, datasetColumns)
dataset = MyPositiveDataset(dataDir, indices, tokenizer, datasetColumns)
negdataset = MyNegativeDataset(dataDir, indices, tokenizer, datasetColumns)
randomdataset = MyRandomDataset(dataDir, indices, tokenizer, datasetColumns)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn = collate_fn)
neg_loader = DataLoader(negdataset, batch_size=16, shuffle=True, collate_fn = collate_fn, drop_last=True)
random_loader = DataLoader(randomdataset, batch_size=16, shuffle=True, collate_fn = collate_fn)
paired_loader = DataLoader(paireddataset, batch_size=16, shuffle=True, collate_fn = paired_collate_fn)
def sim(x, y):
    return -((x-y)**2).sum(dim=-1)
def criterion(u, v, rand_u, rand_v):
    return sim(u,v).mean() - torch.log(sim(u,v).exp().sum() + sim(u,rand_v).exp().sum() + sim(rand_u,v).exp().sum() + sim(rand_u,rand_v).exp().sum())

CE = nn.CrossEntropyLoss()

numEpochs = 200
pairedEpochs = 20
model = Model(numEmbeddings).cuda()

paired_optimizer = AdamW([{'params': model.transformer.parameters(), 'lr':3e-5},
                  {'params':model.fc_paired.parameters()}], lr=1e-3)
paired_scheduler = get_linear_schedule_with_warmup(paired_optimizer, num_warmup_steps=0, num_training_steps = len(paired_loader) * pairedEpochs)
for epoch in range(pairedEpochs):
    for (x, label) in paired_loader:
        pred = model.forward_paired(x)
        loss = CE(pred, label.cuda().long())
        paired_optimizer.zero_grad()
        loss.backward()
        paired_optimizer.step()
        paired_scheduler.step()


optimizer = AdamW(model.fc.parameters(), lr=1e-3)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = len(train_loader) * numEpochs)
for epoch in range(numEpochs):
    negdataloader_iterator = iter(neg_loader)
    for (x, y, label), (randx, randy, rand_label) in zip(train_loader, random_loader):
        embeddings_x, embeddings_y = model.forward_unpaired(x, y)
        random_embeddings_x, random_embeddings_y = model.forward_unpaired(randx, randy)
        label = label.long().cuda()
        classification_loss = 0
        contrastive_loss = 0
        diversity_loss_x = 0
        diversity_loss_y = 0
        B = embeddings_x[0].shape[0]
        for i in range(numEmbeddings):
            u = embeddings_x[i]
            v = embeddings_y[i]
            rand_u = random_embeddings_x[i][np.random.permutation(B)]
            rand_v = random_embeddings_y[i][np.random.permutation(B)]
            contrastive_loss = contrastive_loss - criterion(u, v, rand_u, rand_v)
        loss = (contrastive_loss/numEmbeddings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

model.eval()
torch.set_grad_enabled(False)
xEmbeddings = []
for i in range(numEmbeddings):
    xEmbeddings.append([])
for idx in range(0, len(dataset.xexamples), 16):
    x = dataset.xexamples[idx:idx+16]
    x = _tensorize_batch([torch.tensor(i, dtype=torch.long) for i in x]).cuda()
    attn_x = (x != tokenizer.pad_token_id).float().cuda()
    embed_x = model.transformer(input_ids = x, attention_mask=attn_x)[0]
    embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
    for i in range(numEmbeddings):
        xEmbeddings[i].extend(model.fc[i](embed_x))
for i in range(numEmbeddings):
    xEmbeddings[i] = torch.stack(xEmbeddings[i])
xEmb = torch.stack(xEmbeddings).cpu().numpy()

xEmbeddings = []
for i in range(numEmbeddings):
    xEmbeddings.append([])
for idx in range(0, len(dataset.yexamples), 16):
    x = dataset.yexamples[idx:idx+16]
    x = _tensorize_batch([torch.tensor(i, dtype=torch.long) for i in x]).cuda()
    attn_x = (x != tokenizer.pad_token_id).float().cuda()
    embed_x = model.transformer(input_ids = x, attention_mask=attn_x)[0]
    embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
    for i in range(numEmbeddings):
        xEmbeddings[i].extend(model.fc[i](embed_x))
for i in range(numEmbeddings):
    xEmbeddings[i] = torch.stack(xEmbeddings[i])
yEmb = torch.stack(xEmbeddings).cpu().numpy()
del xEmbeddings
if args.norm:
    xEmb = xEmb/np.linalg.norm(xEmb, axis=-1)[:, :, np.newaxis]
    yEmb = yEmb/np.linalg.norm(yEmb, axis=-1)[:, :, np.newaxis]
Ds = []
Is = []
if args.method == 'l2':
    func = faiss.IndexFlatL2
elif args.method == 'l1':
    func = lambda x: faiss.IndexFlat(x, faiss.METRIC_L1)
elif args.method == 'cosine':
    func = faiss.IndexFlatIP
else:
    exit(1)
test_indices = np.loadtxt(dataDir + 'test.txt', delimiter=',').astype(int).tolist()
indices = indices.tolist() + test_indices
indices = list(map(tuple, indices))
seen = set()
seen_add = seen.add
out = []
for i in range(numEmbeddings):
    index = func(768)
    index.add(xEmb[i])
    D, I = index.search(yEmb[i], k)
    if args.method == 'cosine':
        D = -D
    Ds.append(D)
    Is.append(I)
Ds = np.stack(Ds)
Is = np.stack(Is)
sorted_indices = np.unravel_index(np.argsort(Ds.ravel()), Ds.shape)
Y = sorted_indices[1]
X = Is[tuple([i for i in zip(sorted_indices)])][0]
out = np.stack([X, Y], -1)
out = list(map(tuple, out.tolist()))
seen = set()
seen_add = seen.add
out = [x for x in out if not (x in seen or seen_add(x))][:k*numSampleRetrieve]
np.savetxt(dataDir + args.indices+'CandidateSet'+str(len_indices), np.array(out), fmt='%d,%d')
indices_out = [i for i in out if i not in indices]
def evaluate(matches, ranklist):
    myset = set(matches)
    count = 0
    last = -1
    for idx, i in enumerate(ranklist):
        if i in myset:
            count += 1
    print("Recall", count/len(matches), "Num Matches", len(matches), "Retrieved", count, "All", len(ranklist))
matches = dataset.matches
matches = list(map(tuple, matches))
evaluate(matches, out)
indices = np.array(indices_out)
model.eval()
paireddataset.indices = indices
paireddataset.mode = 'train'
paireddataset.labels = np.zeros_like(indices)
loader = DataLoader(paireddataset, batch_size=16, shuffle=False, collate_fn = paired_collate_fn)
out = []
for (x, _) in loader:
    out.extend(torch.softmax(model.forward_paired(x), -1).cpu().numpy())
out = np.array(out)
entropy = -(out[:, 0]*np.log(out[:, 0]) + out[:, 1]*np.log(out[:, 1]))
ent_indices = np.argsort(entropy)[::-1]
data = indices[ent_indices[:LB]]
np.savetxt(dataDir + args.out, data, fmt='%d,%d')