import copy
import os
from typing import Counter, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.trainer.supporters import CombinedLoader, CycleIterator

class KGDataset(Dataset):
    def __init__(self, name: str, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.entity2idx = {}
        self.relation2idx = {}
        self.entity_freq = None
        self._trn, self._val, self._tst = None, None, None
        self.mode = 'tail'

        self._load_dataset(name)
        self._build_vocab()
        self._map()

    def _load_dataset(self, name: str):
        data_folder = f"./data/{name}/"
        file_list = os.listdir(data_folder)
        # TODO(@AngusHuang17): there are two methods to process valid and test data: raw and filter. 
        # raw means keeping original datasets, filter means remove triplets with same input in train data while has different traget
        # e.g. Train: (e1,r1,e2), Valid: (e1,r1,e3)
        # here we use raw method
        for fname in file_list:
            path = os.path.join(data_folder, fname)
            if 'train' in fname:
                self._trn = pd.read_csv(path, sep='\t', header=None, names=['head', 'relation', 'tail'])
            elif 'valid' in fname:
                self._val = pd.read_csv(path, sep='\t', header=None, names=['head', 'relation', 'tail'])
            elif 'test' in fname:
                self._tst = pd.read_csv(path, sep='\t', header=None, names=['head', 'relation', 'tail'])

    def _build_vocab(self):
        all_entity = Counter({})
        all_relation = Counter({})
        for d in [self._trn, self._val, self._tst]:
            entity_set = Counter(d['head'])
            entity_set.update(Counter(d['tail']))
            all_entity.update(entity_set)
            all_relation.update(Counter(d['relation']))
        self.entity2idx = {e: i+1 for i,e in enumerate(all_entity.keys())}
        self.entity2idx['<pad>'] = 0

        self.relation2idx = {r: i+1 for i,r in enumerate(all_relation.keys())}
        self.relation2idx['<pad>'] = 0

        entity_freq = [0,] + [all_entity[k] for k in all_entity.keys()]
        self.entity_freq = torch.tensor(entity_freq)

    def _map(self):
        def map_id(data):
            data['head'] = data['head'].map(self.entity2idx)
            data['tail'] = data['tail'].map(self.entity2idx)
            data['relation'] = data['relation'].map(self.relation2idx)
            return torch.LongTensor(data.to_numpy())
        self._trn, self._val, self._tst = map_id(self._trn), map_id(self._val), map_id(self._tst)
        
    def build(self):
        trn_data = copy.copy(self)
        trn_data.data = self._trn
        val_data = copy.copy(self)
        val_data.data = self._val
        tst_data = copy.copy(self)
        tst_data.data = self._tst
        return trn_data, val_data, tst_data

    def trans_mode(self):
        if self.mode == 'head':
            self.mode = 'tail'
        else:
            self.mode = 'head'

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        batch = {}
        if self.mode == 'head': # return head as target
            batch['relation'] = self.data[idx][1]
            batch['tail'] = self.data[idx][2]
            batch['target'] = self.data[idx][0]
        else:   # return tail as target
            batch['relation'] = self.data[idx][1]
            batch['head'] = self.data[idx][0]
            batch['target'] = self.data[idx][2]
        return batch
        
    def loader(self, batch_size, shuffle, num_workers):
        head_loader = DataLoader(self, batch_size, shuffle, num_workers=num_workers)
        data_T = copy.copy(self)
        data_T.trans_mode()
        tail_loader = DataLoader(data_T, batch_size, shuffle, num_workers=num_workers)
        # return CombinedLoader([head_loader, tail_loader])
        return AlternatingLoaders([head_loader, tail_loader])

    def train_loader(self, batch_size, num_workers):
        return self.loader(batch_size, True, num_workers)

    def eval_loader(self, batch_size, num_workers):
        return self.loader(batch_size, False, num_workers)

    @property
    def num_items(self):
        return len(self.entity2idx)   # padding included

    @property
    def item_freq(self):
        return self.entity_freq

    @property
    def num_relations(self):
        return len(self.relation2idx)



class AlternatingLoaders(object):
    def __init__(self, loaders) -> None:
        r"""
        The first loader is the main loader.
        """
        self.loaders = []
        for l in loaders:
            self.loaders.append(self.one_shot_iterator(l))
            # self.loaders.append(iter(l))  # bug caused by lightning, early stop error
        self._length = 0
        self.step = 0
        for l in loaders:
            self._length += len(l)

    def __len__(self):
        return self._length

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.step % len(self.loaders)
        batch = next(self.loaders[idx])
        self.step += 1
        if self.step == self._length:
            self.step = 0
            # raise StopIteration
        return batch

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


if __name__ == '__main__':
    config = {
        "max_seq_len" : 20
    }
    dataset = KGDataset('fb15k', config)
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