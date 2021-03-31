import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from transformers import AdamW
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

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--model_init", type=str, default='roberta-base')
parser.add_argument("--indices", type=str, default=['initial_indices.txt'], nargs='*', action='store')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--data", type=str, default='')
args = parser.parse_args()
datasetColumns = None
dataDir = None
if args.data == 'walmart_amazon_exp':
    datasetColumns = ['title', 'category', 'brand', 'modelno', 'price']
    dataDir = 'data/walmart_amazon_exp/'
elif args.data == 'amazon_google_exp':
    datasetColumns = ['title', 'manufacturer', 'price']
    dataDir = 'data/amazon_google_exp/'
elif args.data == 'dblp_acm_exp':
    datasetColumns = ['title', 'authors', 'venue', 'year']
    dataDir = 'data/dblp_acm_exp/'
elif args.data == 'abt_buy_exp':
    datasetColumns = ['name', 'description', 'price']
    dataDir = 'data/abt_buy_exp/'
elif args.data == 'dblp_scholar_exp':
    datasetColumns = ['title', 'authors', 'venue', 'year']
    dataDir = 'data/dblp_scholar_exp/'
else:
    exit(1)


indices_all = np.concatenate([np.loadtxt(dataDir + i, delimiter=',') for i in args.indices]).astype(int)
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
#         print(self.transformer.load_state_dict(torch.load(args.model_init+"/pytorch_model.bin"), strict=False))
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

dataset = MyPairedDataset(dataDir, indices_all, tokenizer, datasetColumns)
criterion = nn.CrossEntropyLoss()
numEpochs = 20
for i in range(10):
    out = np.loadtxt(dataDir + args.indices[0] +'CandidateSet'+str(i+1), delimiter=',').astype(int)
    dataset.mode = 'train'
    dataset.indices = indices_all[:256 + (128*i)]
    dataset.labels = np.array([1 if pair in dataset.matches else 0 for pair in dataset.indices.tolist()])
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn = collate_fn)
    
    print("Num Matches:", train_loader.dataset.labels.sum(), "Size", train_loader.dataset.__len__())
    model = Model().cuda()

    optimizer = AdamW(model.parameters(), lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = len(train_loader) * numEpochs)
    for epoch in range(numEpochs):
        for (x, label) in train_loader:
            pred = model(x)
            loss = criterion(pred, label.cuda().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    dataset.indices = out.copy()
    dataset.labels = np.array([1 if i in dataset.matches else 0 for i in dataset.indices.tolist()])
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn = collate_fn)
    labels = []
    preds = []
    model.eval()
    with torch.no_grad():
        for (x, label) in test_loader:
            pred = model(x)
            preds.extend(pred.cpu().argmax(-1))
            labels.extend(label.cpu())
    tp = 0
    fp = 0
    for i in range(len(labels)):
        if preds[i] == 1:
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
    fn = len(dataset.matches) - tp
    tn = len(dataset.tableA)*len(dataset.tableB) - fp
    print("True Positive", tp, "False Negative", fn, "False Positive", fp, "True Negative", tn)
    f1 = tp/(tp+(0.5*(fp+fn)))
    print("F1", f1)