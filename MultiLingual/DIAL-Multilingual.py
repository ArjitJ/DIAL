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
from transformers import AdamW, RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, get_linear_schedule_with_warmup, BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from dataloader_all import MyDataset
from test_dataloader import MyTestDataset
from itertools import cycle
from statistics import mean


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', max_len=512)

parser = argparse.ArgumentParser()
parser.add_argument("--numEmbeddings", type=int, default=3)
parser.add_argument("--model_init", type=str, default='roberta-base')
parser.add_argument("--indices", type=str, default='initial_50_1.txt')
parser.add_argument("--out", type=str, default='dup_multiple_embeddings_indices.txt')
parser.add_argument("--data", type=str, default='')
parser.add_argument("--norm", action='store_true')
parser.add_argument("--method", type=str, choices=['l2', 'l1', 'cosine'], default='l2')

args = parser.parse_args()
k=3
LB = 128
BATCH_SIZE = 16
MASK_PROB = 0.5
numSampleRetrieve = 100000

config = BertConfig.from_pretrained('bert-base-multilingual-cased', num_hidden_layers=6)
torch.manual_seed(0)
np.random.seed(0)
indices = np.loadtxt('data/'+ args.indices, delimiter=',').astype(int)
numEmbeddings = args.numEmbeddings
len_indices = len(indices)//128

class MyPairedDataset(MyDataset):
    def __init__(self, indices, tokenizer):
        super(MyPairedDataset, self).__init__(indices, tokenizer)
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = self.indices[i, 0]
        y = self.indices[i, 1]
        lbl = self.labels[i]
        xexample = self.xexamples[x][1:-1]
        yexample = self.yexamples[y][1:-1]
        lenA, lenB = len(xexample), len(yexample)
        if lenA + lenB > 508:
            lenA, lenB = min(max(254, 508-lenB), lenA), min(max(254, 508-lenA), lenB)
        input_tokens = []
        input_tokens += [tokenizer.cls_token_id]
        input_tokens += xexample[:lenA]
        input_tokens += [tokenizer.sep_token_id]
        input_tokens += yexample[:lenB]
        input_tokens += [tokenizer.sep_token_id]
        
        
        return torch.tensor(input_tokens, dtype=torch.long), lbl

class MyPositiveDataset(MyDataset):
    def __init__(self, indices, tokenizer):
        super(MyPositiveDataset, self).__init__(indices, tokenizer)
        mask = self.labels == 1
        self.indices = self.indices[mask]
        self.labels = self.labels[mask]

class MyNegativeDataset(MyDataset):
    def __init__(self, indices, tokenizer):
        super(MyNegativeDataset, self).__init__(indices, tokenizer)
        mask = self.labels == 0
        self.indices = self.indices[mask]
        self.labels = self.labels[mask]
        
class MyRandomDataset(MyDataset):
    def __init__(self, indices, tokenizer):
        super(MyRandomDataset, self).__init__(indices, tokenizer)
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
        self.transformer = BertModel(config=config).from_pretrained('bert-base-multilingual-cased', config=config)
        self.transformer.train()
        self.fc = nn.ModuleList([RobertaClassificationHead() for i in range(numEmbeddings)])
        self.fc_paired = PairedRobertaClassificationHead()
        for param in self.transformer.parameters():
            param.requires_grad = False
        
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
        with torch.no_grad():
            x = x.cuda()
            attn_x = (x != tokenizer.pad_token_id).float().cuda()
            embed_x = self.transformer(input_ids = x, attention_mask=attn_x)[0]
            embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
        return self.fc_paired(embed_x)

paireddataset = MyPairedDataset(indices, tokenizer)
dataset = MyPositiveDataset(indices, tokenizer)
randomdataset = MyRandomDataset(indices, tokenizer)
testdataset = MyTestDataset(tokenizer)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate_fn)
random_loader = DataLoader(randomdataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate_fn)
paired_loader = DataLoader(paireddataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = paired_collate_fn)
def sim(x, y):
    return -((x-y)**2).sum(dim=-1)
def criterion(u, v, rand_u, rand_v):
    return sim(u,v).mean() - torch.log(sim(u,v).exp().sum() + sim(u,rand_v).exp().sum() + sim(rand_u,v).exp().sum() + sim(rand_u,rand_v).exp().sum())

CE = nn.CrossEntropyLoss()

numEpochs = 200
pairedEpochs = 20
model = Model(numEmbeddings).cuda()

paired_optimizer = AdamW([{'params':model.fc_paired.parameters()}], lr=1e-3)
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
for idx in range(0, len(testdataset.xexamples), BATCH_SIZE):
    x = dataset.xexamples[idx:idx+BATCH_SIZE]
    x = _tensorize_batch([torch.tensor(i, dtype=torch.long) for i in x]).cuda()
    attn_x = (x != tokenizer.pad_token_id).float().cuda()
    embed_x = model.transformer(input_ids = x, attention_mask=attn_x)[0]
    embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
    for i in range(numEmbeddings):
        xEmbeddings[i].extend((model.fc[i](embed_x)).cpu())
for i in range(numEmbeddings):
    xEmbeddings[i] = torch.stack(xEmbeddings[i])
xEmb = torch.stack(xEmbeddings).cpu().numpy()

xEmbeddings = []
for i in range(numEmbeddings):
    xEmbeddings.append([])
for idx in range(0, len(testdataset.yexamples), BATCH_SIZE):
    x = dataset.yexamples[idx:idx+BATCH_SIZE]
    x = _tensorize_batch([torch.tensor(i, dtype=torch.long) for i in x]).cuda()
    attn_x = (x != tokenizer.pad_token_id).float().cuda()
    embed_x = model.transformer(input_ids = x, attention_mask=attn_x)[0]
    embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
    for i in range(numEmbeddings):
        xEmbeddings[i].extend((model.fc[i](embed_x)).cpu())
for i in range(numEmbeddings):
    xEmbeddings[i] = torch.stack(xEmbeddings[i])
yEmb = torch.stack(xEmbeddings).cpu().numpy()
del xEmbeddings
if args.norm:
    xEmb = xEmb/np.linalg.norm(xEmb, axis=-1)[:, :, np.newaxis]
    yEmb = yEmb/np.linalg.norm(yEmb, axis=-1)[:, :, np.newaxis]
Ds = []
Is = []
k = 3
if args.method == 'l2':
    func = faiss.IndexFlatL2
elif args.method == 'l1':
    func = lambda x: faiss.IndexFlat(x, faiss.METRIC_L1)
elif args.method == 'cosine':
    func = faiss.IndexFlatIP
else:
    exit(1)
testout = []
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
testout = np.stack([X, Y], -1)
testout = list(map(tuple, testout.tolist()))
seen = set()
seen_add = seen.add
testout = [x for x in testout if not (x in seen or seen_add(x))][:6000]
np.savetxt('data/' + args.indices+'CandidateSetTEST'+str(len_indices), np.array(testout), fmt='%d,%d')

xEmbeddings = []
for i in range(numEmbeddings):
    xEmbeddings.append([])
for idx in range(0, len(dataset.xexamples), BATCH_SIZE):
    x = dataset.xexamples[idx:idx+BATCH_SIZE]
    x = _tensorize_batch([torch.tensor(i, dtype=torch.long) for i in x]).cuda()
    attn_x = (x != tokenizer.pad_token_id).float().cuda()
    embed_x = model.transformer(input_ids = x, attention_mask=attn_x)[0]
    embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
    for i in range(numEmbeddings):
        xEmbeddings[i].extend((model.fc[i](embed_x)).cpu())
for i in range(numEmbeddings):
    xEmbeddings[i] = torch.stack(xEmbeddings[i])
xEmb = torch.stack(xEmbeddings).cpu().numpy()

xEmbeddings = []
for i in range(numEmbeddings):
    xEmbeddings.append([])
for idx in range(0, len(dataset.yexamples), BATCH_SIZE):
    x = dataset.yexamples[idx:idx+BATCH_SIZE]
    x = _tensorize_batch([torch.tensor(i, dtype=torch.long) for i in x]).cuda()
    attn_x = (x != tokenizer.pad_token_id).float().cuda()
    embed_x = model.transformer(input_ids = x, attention_mask=attn_x)[0]
    embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
    for i in range(numEmbeddings):
        xEmbeddings[i].extend((model.fc[i](embed_x)).cpu())
for i in range(numEmbeddings):
    xEmbeddings[i] = torch.stack(xEmbeddings[i])
yEmb = torch.stack(xEmbeddings).cpu().numpy()
del xEmbeddings
if args.norm:
    xEmb = xEmb/np.linalg.norm(xEmb, axis=-1)[:, :, np.newaxis]
    yEmb = yEmb/np.linalg.norm(yEmb, axis=-1)[:, :, np.newaxis]
Ds = []
Is = []
k = 3
if args.method == 'l2':
    func = faiss.IndexFlatL2
elif args.method == 'l1':
    func = lambda x: faiss.IndexFlat(x, faiss.METRIC_L1)
elif args.method == 'cosine':
    func = faiss.IndexFlatIP
else:
    exit(1)
indices = indices.tolist()
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
np.savetxt('data/' + args.indices+'CandidateSet'+str(len_indices), np.array(out), fmt='%d,%d')
indices_out = [i for i in out if i not in indices]
def evaluate(matches, ranklist):
    myset = set(matches)
    count = 0
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
loader = DataLoader(paireddataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn = paired_collate_fn)
out = []
for (x, _) in loader:
    out.extend(torch.softmax(model.forward_paired(x), -1).cpu().numpy())
out = np.array(out)
entropy = -(out[:, 0]*np.log(out[:, 0]) + out[:, 1]*np.log(out[:, 1]))
ent_indices = np.argsort(entropy)[::-1]
data = indices[ent_indices[:LB]]
np.savetxt('data/'+args.out, data, fmt='%d,%d')