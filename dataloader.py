import torch
import pandas as pd
import numpy as np
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, indices, tokenizer, datasetColumns):
        self.indices = np.copy(indices)
        self.tokenizer = tokenizer
        self.matches = np.array(pd.read_csv(file_path + 'matches.csv', header=None)).tolist()
        self.tableA = pd.read_csv(file_path+'tableA.csv').fillna("")
        self.tableB = pd.read_csv(file_path+'tableB.csv').fillna("")
        
        self.xexamples = []
        for i in range(self.tableA.shape[0]):
            input_ids = []
            input_ids += [self.tokenizer.cls_token_id]
            for j, colname in enumerate(datasetColumns):
                encoding = self.tokenizer(str(self.tableA[colname][i]), add_special_tokens=False)["input_ids"]
                if len(encoding)>0:
                    input_ids += [1437]
                    input_ids += encoding
            input_ids += [self.tokenizer.sep_token_id]
            self.xexamples.append(input_ids)
            
        self.yexamples = []
        for i in range(self.tableB.shape[0]):
            input_ids = []
            input_ids += [self.tokenizer.cls_token_id]
            for j, colname in enumerate(datasetColumns):
                encoding = self.tokenizer(str(self.tableB[colname][i]), add_special_tokens=False)["input_ids"]
                if len(encoding)>0:
                    input_ids += [1437]
                    input_ids += encoding
            input_ids += [self.tokenizer.sep_token_id]
            self.yexamples.append(input_ids)

        self.labels = np.array([1 if i in self.matches else 0 for i in self.indices.tolist()])
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = self.indices[i, 0]
        y = self.indices[i, 1]
        xexample = self.xexamples[x]
        yexample = self.yexamples[y]
        return torch.tensor(xexample, dtype=torch.long),  torch.tensor(yexample, dtype=torch.long),  self.labels[i]