import argparse
from collections import defaultdict
import random
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from transformers import AdamW, RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, get_linear_schedule_with_warmup
from dataloader import MyDataset
config = RobertaConfig(
    vocab_size=50265,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_len=512)

parser = argparse.ArgumentParser()
parser.add_argument("--model_init", type=str, default='roberta-base')
parser.add_argument("--indices", type=str, default='initial_50_1.txt')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--data", type=str, default='')

parser.add_argument("--numEmbeddings", type=int, default=1)
parser.add_argument("--ortho", type=int, default=1)
parser.add_argument("--temperature", type=float, default=1.)
parser.add_argument("--maskprob", type=float, default=0.5)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--out", type=str, default='dup_multiple_embeddings_indices.txt')
parser.add_argument("--extra", type=str, default='')
parser.add_argument("--norm", action='store_true')
parser.add_argument("--method", type=str, choices=['l2', 'l1', 'cosine'], default='l2')

args = parser.parse_args()
datasetColumns = None
dataDir = None
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
elif args.data == 'dblp_scholar':
    datasetColumns = ['title', 'authors', 'venue', 'year']
    dataDir = 'data/dblp_scholar/'
else:
    exit(1)



indices = np.loadtxt(dataDir + args.indices, delimiter=',').astype(int)
out = map(list, list(set(map(tuple, indices.tolist()))))
indices = np.array(list(out)).astype(int)
numEmbeddings = args.numEmbeddings
torch.manual_seed(0)
np.random.seed(0)
len_indices = len(indices)//128

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

class MyPositiveDataset(MyPairedDataset):
    def __init__(self, file_path, indices, tokenizer, datasetColumns):
        super(MyPositiveDataset, self).__init__(file_path, indices, tokenizer, datasetColumns)
        mask = self.labels == 1
        self.indices = self.indices[mask]
        self.labels = self.labels[mask]

class MyRandomDataset(MyPairedDataset):
    def __init__(self, file_path, indices, tokenizer, datasetColumns):
        super(MyRandomDataset, self).__init__(file_path, indices, tokenizer, datasetColumns)
        self.indices = self.indices[self.labels == 1]
        
    def __getitem__(self, i):
        x = np.random.randint(len(self.xexamples))
        y = np.random.randint(len(self.yexamples))
        xexample = self.xexamples[x][1:-1]
        yexample = self.yexamples[y][1:-1]
            
        input_tokens = []
        input_tokens += [tokenizer.cls_token_id]
        input_tokens += xexample
        input_tokens += [tokenizer.sep_token_id]
        input_tokens += [tokenizer.sep_token_id]
        input_tokens += yexample
        input_tokens += [tokenizer.sep_token_id]
        return torch.tensor(input_tokens, dtype=torch.long), 0*self.labels[i]            
        
    
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
    target = [item[1] for item in batch]
    target = torch.Tensor(target)
    
    xbatch = _tensorize_batch(x)
    
    return xbatch, target

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

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
    def __init__(self):
        super(Model, self).__init__()
        self.transformer = RobertaForMaskedLM(config=config)
        self.transformer.load_state_dict(torch.load(args.model_init+"/pytorch_model.bin"), strict=False)
        self.transformer.train()
        self.fc = RobertaClassificationHead()
        for param in self.transformer.roberta.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        x = x.cuda()
        attn_x = (x != tokenizer.pad_token_id).float().cuda()
        embed_x = self.transformer.roberta(input_ids = x, attention_mask=attn_x)[0]
        embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
        return self.fc(embed_x)

dataset = MyPairedDataset(dataDir, indices, tokenizer, datasetColumns)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn = collate_fn)

criterion = nn.CrossEntropyLoss()
numEpochs = 20
model = Model().cuda()

# optimizer = AdamW(model.parameters(), lr=3e-5)
optimizer = AdamW([{'params': model.transformer.parameters(), 'lr':3e-5},
                  {'params':model.fc.parameters()},], lr=1e-3)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = len(train_loader) * numEpochs)

for epoch in range(numEpochs):
    for (x, label) in train_loader:
        pred = model(x)
        label = label.long().cuda()
        loss = criterion(pred, label.cuda().long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

model.eval()
torch.set_grad_enabled(False)
xEmbeddings = []
for idx in range(0, len(dataset.xexamples), 16):
    x = dataset.xexamples[idx:idx+16]
    x = _tensorize_batch([torch.tensor(i, dtype=torch.long) for i in x]).cuda()
    attn_x = (x != tokenizer.pad_token_id).float().cuda()
    embed_x = model.transformer.roberta(input_ids = x, attention_mask=attn_x)[0]
    embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
    xEmbeddings.extend(embed_x)
xEmb = torch.stack(xEmbeddings).cpu().numpy()

xEmbeddings = []
for idx in range(0, len(dataset.yexamples), 16):
    x = dataset.yexamples[idx:idx+16]
    x = _tensorize_batch([torch.tensor(i, dtype=torch.long) for i in x]).cuda()
    attn_x = (x != tokenizer.pad_token_id).float().cuda()
    embed_x = model.transformer.roberta(input_ids = x, attention_mask=attn_x)[0]
    embed_x = (embed_x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
    xEmbeddings.extend(embed_x)
yEmb = torch.stack(xEmbeddings).cpu().numpy()
del xEmbeddings
if args.norm:
    xEmb = xEmb/np.linalg.norm(xEmb, axis=-1)[:, np.newaxis]
    yEmb = yEmb/np.linalg.norm(yEmb, axis=-1)[:, np.newaxis]
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
index = func(768)
index.add(xEmb)
Ds, Is = index.search(yEmb, k)
if args.method == 'cosine':
    Ds = -Ds
test_indices = np.loadtxt(dataDir + 'test.txt', delimiter=',').astype(int).tolist()
indices = indices.tolist() + test_indices
indices = list(map(tuple, indices))
sorted_indices = np.unravel_index(np.argsort(Ds.ravel()), Ds.shape)
Y = sorted_indices[0]
X = Is[tuple([i for i in zip(sorted_indices)])][0]
out = np.stack([X, Y], -1)
out = list(map(tuple, out.tolist()))
seen = set()
seen_add = seen.add
out = [x for x in out if not (x in seen or seen_add(x))][:3*numSampleRetrieve]
np.savetxt(dataDir + args.indices+'CandidateSet'+str(len_indices), np.array(out), fmt='%d,%d')
indices_out = [i for i in out if i not in indices]
from statistics import mean
def evaluate(matches, ranklist):
    myset = set(matches)
    ranks = []
    count = 0
    last = -1
    for idx, i in enumerate(ranklist):
        if i in myset:
            count += 1
            ranks.append(count/(idx+1))
            last = idx+1
    if count>0:
        print("Recall", count/len(matches), "Num Matches", len(matches), "Retrieved", count, "All", len(ranklist))
    else:
        print("Recall", count/len(matches), "AP", 0)
matches = dataset.matches
matches = list(map(tuple, matches))
evaluate(matches, out)
# random.shuffle(indices_out)
indices = np.array(indices_out)
out = []
model.eval()
temperature = torch.Tensor([args.temperature]).cuda()
dataset.indices = indices
dataset.mode = 'train'
dataset.labels = np.zeros_like(indices)
loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn = collate_fn)
for (x, _) in loader:
    out.extend(torch.softmax(model(x)/temperature, -1).cpu().numpy())
out = np.array(out)
entropy = -(out[:, 0]*np.log(out[:, 0]) + out[:, 1]*np.log(out[:, 1]))
ent_indices = np.argsort(entropy)[::-1]
data = indices[ent_indices[:128]]
np.savetxt(dataDir + args.out, data, fmt='%d,%d')