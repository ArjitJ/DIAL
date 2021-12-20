import torch
import pandas as pd
import numpy as np
import json
from data_utils import get_tokenized


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, indices, tokenizer, datasetColumns):
        self.indices = indices
        self.tokenizer = tokenizer
        xexamples, yexamples, matches = get_tokenized(
            file_path, tokenizer, datasetColumns)
        self.xexamples = xexamples
        self.yexamples = yexamples
        self.matches = matches

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, i):
        x = self.indices[i, 0]
        y = self.indices[i, 1]
        xexample = self.xexamples[x]
        yexample = self.yexamples[y]
        return torch.tensor(xexample, dtype=torch.long),  torch.tensor(yexample, dtype=torch.long),  self.labels[i]
