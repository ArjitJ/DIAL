import torch
import pandas as pd
import numpy as np
import copy
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_len=512)

def get_tokenized(file_path, tokenizer, datasetColumns):
    matches = np.array(pd.read_csv(file_path + 'matches.csv', header=None)).tolist()
    tableA = pd.read_csv(file_path+'tableA.csv').fillna("")
    tableB = pd.read_csv(file_path+'tableB.csv').fillna("")
    xexamples = []
    for i in range(tableA.shape[0]):
        input_ids = []
        input_ids += [tokenizer.cls_token_id]
        for colname in datasetColumns:
            encoding = tokenizer(str(tableA[colname][i]), add_special_tokens=False)["input_ids"]
            if len(encoding)>0:
                input_ids += [1437]
                input_ids += encoding
        input_ids += [tokenizer.sep_token_id]
        xexamples.append(input_ids)
    
    yexamples = []
    for i in range(tableB.shape[0]):
        input_ids = []
        input_ids += [tokenizer.cls_token_id]
        for colname in datasetColumns:
            encoding = tokenizer(str(tableB[colname][i]), add_special_tokens=False)["input_ids"]
            if len(encoding)>0:
                input_ids += [1437]
                input_ids += encoding
        input_ids += [tokenizer.sep_token_id]
        yexamples.append(input_ids)
    return xexamples, yexamples, matches

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, indices, xexamples, yexamples, matches):
        self.indices = np.copy(indices)
        self.matches = copy.deepcopy(matches)
        self.xexamples = xexamples
        self.yexamples = yexamples
        self.labels = np.array([1 if i in self.matches else 0 for i in self.indices.tolist()])
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = self.indices[i, 0]
        y = self.indices[i, 1]
        xexample = self.xexamples[x]
        yexample = self.yexamples[y]
        return torch.tensor(xexample, dtype=torch.long),  torch.tensor(yexample, dtype=torch.long),  self.labels[i]
    
class MyPositiveDataset(MyDataset):
    def __init__(self, indices, xexamples, yexamples, matches):
        super(MyPositiveDataset, self).__init__(indices, xexamples, yexamples, matches)
        mask = self.labels == 1
        self.indices = self.indices[mask]
        self.labels = self.labels[mask]
        
class MyRandomDataset(MyDataset):
    def __init__(self, indices, xexamples, yexamples, matches):
        super(MyRandomDataset, self).__init__(indices, xexamples, yexamples, matches)
        self.indices = self.indices[self.labels == 1]
        
    def __getitem__(self, i):
        x = np.random.randint(len(self.xexamples))
        y = np.random.randint(len(self.yexamples))
        xexample = self.xexamples[x]
        yexample = self.yexamples[y]
        return torch.tensor(xexample, dtype=torch.long),  torch.tensor(yexample, dtype=torch.long),  0*self.labels[i]        

class MyPairedDataset(MyDataset):
    def __init__(self, indices, xexamples, yexamples, matches, dataDir=None):
        super(MyPairedDataset, self).__init__(indices, xexamples, yexamples, matches)
        self.mode = 'train'
        self.test = np.loadtxt(dataDir + 'test.txt', delimiter=',').astype(int)
        
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

def tensorize_batch(examples):
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)
    else:
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

def collate_fn(batch):
    x, y, target = zip(*batch)
    target = torch.Tensor(target)
    x = tensorize_batch(x)
    y = tensorize_batch(y)
    return x, y, target

def paired_collate_fn(batch):
    x, target = zip(*batch)
    target = torch.Tensor(target)
    x = tensorize_batch(x)
    return x, target

def get_data_parameters(data):
    k=3
    datasetColumns = None
    dataDir = None
    numSampleRetrieve = None
    if data == 'walmart_amazon_exp':
        datasetColumns = ['title', 'category', 'brand', 'modelno', 'price']
        dataDir = 'data/walmart_amazon_exp/'
        numSampleRetrieve = 22074
    elif data == 'amazon_google_exp':
        datasetColumns = ['title', 'manufacturer', 'price']
        dataDir = 'data/amazon_google_exp/'
        numSampleRetrieve = 3226
    elif data == 'dblp_acm_exp':
        datasetColumns = ['title', 'authors', 'venue', 'year']
        dataDir = 'data/dblp_acm_exp/'
        numSampleRetrieve = 2294
    elif data == 'abt_buy_exp':
        datasetColumns = ['name', 'description', 'price']
        dataDir = 'data/abt_buy_exp/'
        numSampleRetrieve = 1091
        k=20
    elif data == 'dblp_scholar_exp':
        datasetColumns = ['title', 'authors', 'venue', 'year']
        dataDir = 'data/dblp_scholar_exp/'
        numSampleRetrieve = 64263
    return datasetColumns, dataDir, numSampleRetrieve, k