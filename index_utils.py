import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_utils import tensorize_batch
BATCH_SIZE = 256

def parse_index(method):
    if method == 'l2':
        func = faiss.IndexFlatL2
    elif method == 'l1':
        func = lambda x: faiss.IndexFlat(x, faiss.METRIC_L1)
    elif method == 'cosine':
        func = faiss.IndexFlatIP
    return func

def evaluate(matches, retrieved_pairs):
    myset = set(map(tuple, matches))
    count = 0
    for i in retrieved_pairs:
        if i in myset:
            count += 1
    print("Recall", count/len(matches), "Num Matches", len(matches), "Retrieved", count, "All", len(retrieved_pairs))

def getEmbeddings(lst, tokenizer, model, numEmbeddings, norm):
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
    emb = torch.stack(embeddings).cpu().numpy()
    if norm:
        emb = emb/np.linalg.norm(emb, axis=-1)[:, :, np.newaxis]
    return emb