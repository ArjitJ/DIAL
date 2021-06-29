import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
import pandas as pd
from transformers import AdamW, RobertaConfig, RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup, logging
from torch.utils.data import DataLoader
from data_utils import *
from model_utils import *
logging.set_verbosity_error()

tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_len=512)

parser = argparse.ArgumentParser()
parser.add_argument("--numEmbeddings", type=int, default=3)
parser.add_argument("--indices", type=str, default='', required=True)
parser.add_argument("--out", type=str, default='', required=True)
parser.add_argument("--data", type=str, default='', required=True)
parser.add_argument("--norm", action='store_true')
parser.add_argument("--method", type=str, choices=['l2', 'l1', 'cosine'], default='l2')

args = parser.parse_args()
datasetColumns, dataDir, numSampleRetrieve, k = get_data_parameters(args.data)
LB = 128
BATCH_SIZE = 16

numEpochs = 200
pairedEpochs = 20
torch.manual_seed(0)
np.random.seed(0)
indices = np.loadtxt(dataDir + args.indices, delimiter=',').astype(int)
numEmbeddings = args.numEmbeddings
len_indices = len(indices)//LB


    
xexamples, yexamples, matches = get_tokenized(dataDir, tokenizer, datasetColumns)
paireddataset = MyPairedDataset(indices, xexamples, yexamples, matches, dataDir)
dataset = MyPositiveDataset(indices, xexamples, yexamples, matches)
randomdataset = MyRandomDataset(indices, xexamples, yexamples, matches)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate_fn)
random_loader = DataLoader(randomdataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate_fn)
paired_loader = DataLoader(paireddataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = paired_collate_fn)


CE = nn.CrossEntropyLoss()

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
    for (x, y, _), (randx, randy, _) in zip(train_loader, random_loader):
        embeddings_x, embeddings_y = model.forward_unpaired(x, y)
        random_embeddings_x, random_embeddings_y = model.forward_unpaired(randx, randy)
        contrastive_loss = 0
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

def getEmbeddings(lst):
    embeddings = []
    for i in range(numEmbeddings):
        embeddings.append([])
    for idx in range(0, len(lst), BATCH_SIZE):
        x = lst[idx:idx+BATCH_SIZE]
        x = tensorize_batch([torch.tensor(i, dtype=torch.long) for i in x]).cuda()
        attn_x = (x != tokenizer.pad_token_id).float().cuda()
        x = model.transformer(input_ids = x, attention_mask=attn_x)[0]
        x = (x*attn_x.unsqueeze(-1)).sum(1)/attn_x.unsqueeze(-1).sum(1)
        for i in range(numEmbeddings):
            embeddings[i].extend(model.fc[i](x))
    for i in range(numEmbeddings):
        embeddings[i] = torch.stack(embeddings[i])
    return torch.stack(embeddings).cpu().numpy()

xEmb = getEmbeddings(xexamples)
yEmb = getEmbeddings(yexamples)
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
    out.extend(torch.softmax(model.forward_paired(x), -1))
out = torch.stack(out).cpu().numpy()
entropy = -(out[:, 0]*np.log(out[:, 0]) + out[:, 1]*np.log(out[:, 1]))
ent_indices = np.argsort(entropy)[::-1]
data = indices[ent_indices[:LB]]
np.savetxt(dataDir + args.out, data, fmt='%d,%d')
