import torch
import pandas as pd
import numpy as np
import json
class MyTestDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tableA = json.load(open('localization-xml-mt/data/ende/ende_en_dev.json'))['text']
        self.tableB = json.load(open('localization-xml-mt/data/ende/ende_de_dev.json'))['text']
        self.keys = list(self.tableA.keys())
        self.xexamples = []
        for i in self.keys:
            input_ids = [self.tokenizer.cls_token_id, *self.tokenizer(self.tableA[i], add_special_tokens=False)["input_ids"][:510], self.tokenizer.sep_token_id]
            self.xexamples.append(input_ids)
            
        self.yexamples = []
        for i in self.keys:
            input_ids = [self.tokenizer.cls_token_id, *self.tokenizer(self.tableB[i], add_special_tokens=False)["input_ids"][:510], self.tokenizer.sep_token_id]
            self.yexamples.append(input_ids)

        self.matches = [[i, i] for i in range(len(self.xexamples))]
        
    def __len__(self):
        pass

    def __getitem__(self, i):
        pass
