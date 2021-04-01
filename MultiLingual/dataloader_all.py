import torch
import pandas as pd
import numpy as np
import json
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, indices, tokenizer):
        self.indices = indices
        self.tokenizer = tokenizer
        self.tableA = json.load(open('localization-xml-mt/data/ende/ende_en_train.json'))['text']
        self.tableB = json.load(open('localization-xml-mt/data/ende/ende_de_train.json'))['text']
        self.keys = list(self.tableA.keys())
        self.xexamples = []
        for i in self.keys:
            input_ids = [self.tokenizer.cls_token_id, *self.tokenizer(self.tableA[i], add_special_tokens=False)["input_ids"][:510], self.tokenizer.sep_token_id]
            self.xexamples.append(input_ids)
            
        self.yexamples = []
        for i in self.keys:
            input_ids = [self.tokenizer.cls_token_id, *self.tokenizer(self.tableB[i], add_special_tokens=False)["input_ids"][:510], self.tokenizer.sep_token_id]
            self.yexamples.append(input_ids)

        self.labels = np.array([1 if (i[0] == i[1]) else 0 for i in self.indices.tolist()])
        self.matches = [[i, i] for i in range(len(self.xexamples))]
    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, i):
        x = self.indices[i, 0]
        y = self.indices[i, 1]
        xexample = self.xexamples[x]
        yexample = self.yexamples[y]
        return torch.tensor(xexample, dtype=torch.long),  torch.tensor(yexample, dtype=torch.long),  self.labels[i]