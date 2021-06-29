# imports
import argparse
import torch
import torch.nn as nn
import faiss
import numpy as np
from transformers import AdamW, RobertaConfig, RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup, logging
from torch.utils.data import DataLoader
from data_utils import *
from model_utils import *
from index_utils import *
logging.set_verbosity_error()
torch.manual_seed(0)
np.random.seed(0)

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--numEmbeddings", type=int, default=3)
parser.add_argument("--indices", type=str, default='', required=True)
parser.add_argument("--out", type=str, default='')
parser.add_argument("--data", type=str, default='', required=True)
parser.add_argument("--norm", action='store_true')
parser.add_argument("--method", type=str, choices=['l2', 'l1', 'cosine'], default='l2')
args = parser.parse_args()

# hyperparameters and problem configuration
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_len=512)
datasetColumns, dataDir, numSampleRetrieve, k = get_data_parameters(args.data)
LB = 128
BATCH_SIZE = 16
numEpochs = 200
pairedEpochs = 20
indices = np.loadtxt(dataDir + args.indices, delimiter=',').astype(int)
numEmbeddings = args.numEmbeddings
CE = nn.CrossEntropyLoss()

# tokenizing lists R and S
R_tokenized, S_tokenized, matches = get_tokenized(dataDir, tokenizer, datasetColumns)
test_indices = np.loadtxt(dataDir + 'test.txt', delimiter=',').astype(int).tolist()

for active_learning_round in range(10):
    model = Model(numEmbeddings).cuda()
    len_indices = len(indices)//LB
    paireddataset = MyPairedDataset(indices, R_tokenized, S_tokenized, matches, dataDir)
    dataset = MyPositiveDataset(indices, R_tokenized, S_tokenized, matches)
    randomdataset = MyRandomDataset(indices, R_tokenized, S_tokenized, matches)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate_fn)
    random_loader = DataLoader(randomdataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate_fn)
    paired_loader = DataLoader(paireddataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = paired_collate_fn)

    # train matcher
    paired_optimizer = AdamW([{'params': model.transformer.parameters(), 'lr':3e-5}, {'params':model.fc_paired.parameters()}], lr=1e-3)
    paired_scheduler = get_linear_schedule_with_warmup(paired_optimizer, num_warmup_steps=0, num_training_steps = len(paired_loader) * pairedEpochs)
    for epoch in range(pairedEpochs):
        for (x, label) in paired_loader:
            pred = model.forward_paired(x)
            loss = CE(pred, label.cuda().long())
            paired_optimizer.zero_grad()
            loss.backward()
            paired_optimizer.step()
            paired_scheduler.step()

    # train committee
    optimizer = AdamW(model.fc.parameters(), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = len(train_loader) * numEpochs)
    for epoch in range(numEpochs):
        np.random.shuffle(random_loader.dataset.xexamples)
        np.random.shuffle(random_loader.dataset.yexamples)
        for (r_p, s_p, _), (rand_r, rand_s, _) in zip(train_loader, random_loader):
            # single mode: get embeddings for positives and random negatives
            embeddings_r, embeddings_s = model.forward_unpaired(r_p, s_p)
            random_embeddings_r, random_embeddings_s = model.forward_unpaired(rand_r, rand_s)
            contrastive_loss = 0
            B = embeddings_r[0].shape[0]
            for i in range(numEmbeddings):
                E_r_p = embeddings_r[i]
                E_s_p = embeddings_s[i]
                E_rand_r = random_embeddings_r[i][np.random.permutation(B)]
                E_rand_s = random_embeddings_s[i][np.random.permutation(B)]
                contrastive_loss = contrastive_loss - criterion(E_r_p, E_s_p, E_rand_r, E_rand_s)
            loss = (contrastive_loss/numEmbeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    torch.set_grad_enabled(False)
    model.eval()
    # Shape (N, |R|)
    R_embeddings = getEmbeddings(R_tokenized, tokenizer, model, numEmbeddings, args.norm)
    # Shape (N, |S|)
    S_embeddings = getEmbeddings(S_tokenized, tokenizer, model, numEmbeddings, args.norm)

    # Indexing and Retrieval
    Ds, Is = [], []
    func = parse_index(args.method)
    for i in range(numEmbeddings):
        index = func(768)
        index.add(R_embeddings[i])
        # D and I are of shapes (|S|, k) containing the nearest distances and the corresponding element index in R respectively
        D, I = index.search(S_embeddings[i], k)
        if args.method == 'cosine':
            D = -D
        Ds.append(D)
        Is.append(I)
    Ds, Is = np.stack(Ds), np.stack(Is)

    sorted_indices = np.unravel_index(np.argsort(Ds.ravel()), Ds.shape)
    S = sorted_indices[1]
    R = Is[tuple([i for i in zip(sorted_indices)])][0]
    out = map(tuple, np.stack([R, S], -1).tolist())
    seen = set()
    retrieved_pairs = [x for x in out if not (x in seen or seen.add(x))][:k*numSampleRetrieve]
    np.savetxt(dataDir + args.indices+'CandidateSet'+str(len_indices), np.array(retrieved_pairs), fmt='%d,%d')
    not_allowed = indices.tolist() + test_indices
    candidate_set = np.array([i for i in retrieved_pairs if i not in map(tuple, not_allowed)])

    # Selection
    paireddataset.indices = candidate_set
    paireddataset.mode = 'train'
    paireddataset.labels = np.zeros_like(candidate_set)
    loader = DataLoader(paireddataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn = paired_collate_fn)
    out = []
    for (x, _) in loader:
        out.extend(torch.softmax(model.forward_paired(x), -1))
    out = torch.stack(out).cpu().numpy()
    entropy = -(out[:, 0]*np.log(out[:, 0]) + out[:, 1]*np.log(out[:, 1]))
    ent_indices = np.argsort(entropy)[::-1]
    data = candidate_set[ent_indices[:LB]]
    indices = np.concatenate([indices, data], axis=0)
    evaluate(matches, retrieved_pairs)
    torch.set_grad_enabled(True)
    model.train()