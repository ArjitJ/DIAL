import torch
import pandas as pd
import numpy as np

def get_tokenized(file_path, tokenizer, datasetColumns):
    matches = np.array(pd.read_csv(file_path + 'matches.csv', header=None)).tolist()
    tableA = pd.read_csv(file_path+'tableA.csv').fillna("")
    tableB = pd.read_csv(file_path+'tableB.csv').fillna("")
    xexamples = []
    for i in range(tableA.shape[0]):
        input_ids = []
        input_ids += [tokenizer.cls_token_id]
        for j, colname in enumerate(datasetColumns):
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
        for j, colname in enumerate(datasetColumns):
            encoding = tokenizer(str(tableB[colname][i]), add_special_tokens=False)["input_ids"]
            if len(encoding)>0:
                input_ids += [1437]
                input_ids += encoding
        input_ids += [tokenizer.sep_token_id]
        self.yexamples.append(input_ids)
    return xexamples, yexamples, matches

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, indices, xexamples, yexamples, matches):
        self.indices = np.copy(indices)
        self.matches = np.copy(matches)
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