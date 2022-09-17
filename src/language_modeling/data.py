import os
import copy
import math
from collections import Counter
import torch
import numpy as np

from typing import Dict
from torch.utils.data import Dataset, DataLoader

class LanguageModelDataset(Dataset):
    def __init__(self, name: str, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.word2idx = {}
        self.word_freq = None
        self._max_seq_len = self.config['max_seq_len']
        self._trn, self._val, self._tst = None, None, None

        self._load_dataset(name)
        self._build_vocab()
        self._map()

    def _load_dataset(self, name: str):
        name_l = name.lower()
        data_folder = f"./data/{name}/"
        file_list = os.listdir(data_folder)
        for fname in file_list:
            path = os.path.join(data_folder, fname)
            with open(path, 'r') as f:
                if 'train' in fname:
                    self._trn = f.read()[1:].split(' ')
                elif 'valid' in fname:
                    self._val = f.read()[1:].split(' ')
                elif 'test' in fname:
                    self._tst = f.read()[1:].split(' ')

    def _build_vocab(self):
        words = Counter(self._trn + self._val + self._tst)
        words = {k: v for k, v in sorted(words.items(), key=lambda item: -item[1])}
        word_freq = [0,]
        for i, (k, v) in enumerate(words.items()):
            self.word2idx[k] = i+1
            word_freq.append(v)
        self.word2idx['<pad>'] = 0  # padding_index = 0
        self.word_freq = np.array(word_freq)

    def _map(self):
        self._trn = list(map(lambda x:self.word2idx[x], self._trn))
        self._val = list(map(lambda x:self.word2idx[x], self._val))
        self._tst = list(map(lambda x:self.word2idx[x], self._tst))
        self._trn = torch.LongTensor(np.array(self._trn)) 
        self._val = torch.LongTensor(np.array(self._val))
        self._tst = torch.LongTensor(np.array(self._tst))

    def build(self):
        trn_data = copy.copy(self)
        trn_data.data = self._trn
        val_data = copy.copy(self)
        val_data.data = self._val
        tst_data = copy.copy(self)
        tst_data.data = self._tst
        return trn_data, val_data, tst_data
    
    def __len__(self):
        return math.ceil((len(self.data)-1) / self._max_seq_len)

    def __getitem__(self, idx):
        batch = {}
        start, end = idx*self._max_seq_len, min((idx+1)*self._max_seq_len, len(self.data)-1)
        batch['input'] = self.data[start: end]
        batch['target'] = self.data[start+1: end+1]
        if (end - start) < self._max_seq_len:
            _pad_num = self._max_seq_len - (end-start)
            batch['input'] = torch.nn.functional.pad(batch['input'], (0, _pad_num), 'constant', 0)
            batch['target'] = torch.nn.functional.pad(batch['target'], (0, _pad_num), 'constant', 0)
        return batch

    def loader(self, batch_size, shuffle, num_workers):
        return DataLoader(self, batch_size, shuffle, num_workers=num_workers)

    def train_loader(self, batch_size, num_workers):
        return self.loader(batch_size, True, num_workers)

    def eval_loader(self, batch_size, num_workers):
        return self.loader(batch_size, False, num_workers)

    @property
    def num_items(self):
        return len(self.word2idx)   # padding included

    @property
    def max_seq_len(self):
        return self._max_seq_len
    

if __name__ == '__main__':
    config = {
        "max_seq_len" : 20
    }
    dataset = LanguageModelDataset('penntreebank', config)
    trn, val, tst = dataset.build()
    trn_loader = trn.train_loader(128, 0)
    val_loader = val.eval_loader(128, 0)
    tst_loader = tst.eval_loader(128, 0)
    for batch in trn_loader:
        print(batch)
        break
    for batch in val_loader:
        print(batch)
        break
    for batch in tst_loader:
        print(batch)
        break
    print("End")
