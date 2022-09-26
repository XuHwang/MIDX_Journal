import os
import copy
import random
import torch
import numpy as np

from typing import Dict
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, spmatrix
from sklearn.datasets import load_svmlight_file
from torch.nn.utils.rnn import pad_sequence



def gen_shape(indices, indptr, zero_based=True):
    _min = min(indices)
    if not zero_based:
        indices = list(map(lambda x: x-_min, indices))
    num_cols = max(indices)
    num_rows = len(indptr) - 1
    return (num_rows, num_cols)

def expand_indptr(num_rows_inferred, num_rows, indptr):
    """Expand indptr if inferred num_rows is less than given
    """
    _diff = num_rows - num_rows_inferred
    if _diff > 0:  # Fix indptr as per new shape
        # Data is copied here
        import warnings
        warnings.warn("Header mis-match from inferred shape!")
        return np.concatenate((indptr, np.repeat(indptr[-1], _diff)))
    elif _diff == 0:  # It's fine
        return indptr
    else:
        raise NotImplementedError("Unknown behaviour!")

def ll_to_sparse(X, shape=None, dtype='float32', zero_based=True):
    """Convert a list of list to a csr_matrix; All values are 1.0
    Arguments:
    ---------
    X: list of list of tuples
        nnz indices for each row
    shape: tuple or none, optional, default=None
        Use this shape or infer from data
    dtype: 'str', optional, default='float32'
        datatype for data
    zero_based: boolean or "auto", default=True
        indices are zero based or not

    Returns:
    -------
    X: csr_matrix
    """
    indices = []
    indptr = [0]
    offset = 0
    for item in X:
        if len(item) > 0:
            indices.extend(item)
            offset += len(item)
        indptr.append(offset)
    data = [1.0]*len(indices)
    _shape = gen_shape(indices, indptr, zero_based)
    if shape is not None:
        assert _shape[0] <= shape[0], "num_rows_inferred > num_rows_given"
        assert _shape[1] <= shape[1], "num_cols_inferred > num_cols_given"
        indptr = expand_indptr(_shape[0], shape[0], indptr)
    return csr_matrix(
        (np.array(data, dtype=dtype), np.array(indices), np.array(indptr)),
        shape=shape)

def pad_collate_valid(batch):
    elem = batch[0]
    elem_type = type(elem)
    return elem_type({key: pad_sequence([d[key] for d in batch], batch_first=True) for key in elem})

class ExtremeClassDataset(Dataset):
    def __init__(self, name: str, config: Dict ) -> None:
        super().__init__()
        self.config = config

        self._load_dataset(name)


    def _load_dataset(self, name: str):
        data_folder = f"./data/{name}/"
        file_list = os.listdir(data_folder)

        if len(file_list) == 1:
            # one whole dataset
            fname = file_list[0]
            path = os.path.join(data_folder, fname)
            features, labels, num_feat, num_labels = self._load_from_bow(path)

            self.num_feat = num_feat
            self.num_labels = num_labels

            self._split_data(features, labels, self.config['split_ratio'])

        elif len(file_list) == 2:
            # train.txt & test.txt
            for fname in file_list:
                if 'train' in fname:
                    path = os.path.join(data_folder, fname)
                    features, labels, num_feat, num_labels = self._load_from_bow(path)
                    
                    self.num_feat = num_feat
                    self.num_labels = num_labels
                
                elif 'test' in fname:
                    path = os.path.join(data_folder, fname)
                    features_t, labels_t, num_feat_t, num_labels_t = self._load_from_bow(path)
                
                else:
                    pass
            
            assert (num_feat == num_feat_t) and (num_labels == num_labels_t)
            self._split_data_train(features, labels, features_t, labels_t, self.config['split_ratio'])
    
    def _load_from_bow(self, filename: str, header: bool = True, dtype: str='float32', zero_based:bool=True):
        # Use sklearn load_svmlight_file to read data
        # Return sparse format data
        with open(filename, 'rb') as f:
            _l_shape = None
            if header:
                line = f.readline().decode('utf-8').rstrip("\n")
                line = line.split(" ")
                num_samples, num_feat, num_labels = int(
                    line[0]), int(line[1]), int(line[2])
                _l_shape = (num_samples, num_labels)
            else:
                num_samples, num_feat, num_labels = None, None, None
            features, labels = load_svmlight_file(f, n_features=num_feat, multilabel=True, zero_based=zero_based)
            labels = ll_to_sparse(
            labels, dtype=dtype, zero_based=zero_based, shape=_l_shape)
            nonzeros_rows = labels.getnnz(-1) > 0 # remove empty labels
        return features[nonzeros_rows], labels[nonzeros_rows], num_feat, num_labels

    def _split_data(self, features: spmatrix, labels: spmatrix, split_ratio: list = [0.8, 0.1,0.1]):
        trn, vld, tst = self._split_data_indices(features.shape[0], split_ratio)

        self._trn = (features[trn], labels[trn])
        self._val = (features[vld], labels[vld])
        self._tst = (features[tst], labels[tst])
        self.label_freq =  torch.from_numpy(labels[trn].sum(0)).squeeze()
        self.label_freq = torch.cat((torch.zeros(1), self.label_freq))
    
    def _split_data_train(self, features_x: spmatrix, labels_x: spmatrix, features_y: spmatrix, labels_y: spmatrix, split_ratio: float=0.8):
        if isinstance(split_ratio, float):
            split_ratio = split_ratio if (split_ratio > 0 ) and (split_ratio < 1) else 0.8
        else:
            split_ratio = 0.8
        indices_list = list(range(features_x.shape[0]))
        random.shuffle(indices_list)
        num_trn = int(features_x.shape[0] * split_ratio)
        trn, vld = indices_list[:num_trn], indices_list[num_trn:]

        self._trn = (features_x[trn], labels_x[trn])
        self._val = (features_x[vld], labels_x[vld])
        self._tst = (features_y, labels_y)
        self.label_freq =  torch.from_numpy(features_x[trn].sum(0)).squeeze()
        self.label_freq = torch.cat((torch.zeros(1), self.label_freq))


    def _split_data_indices(self, num_samples:int, ratio:list=None): 
        if isinstance(ratio, list):
            ratio = ratio if (np.sum(ratio) - 1.0 ) < 1e-6 else [0.8, 0.1, 0.1]
        else:
            ratio = [0.8, 0.1, 0.1]
        
        indices_list = list(range(num_samples))
        random.shuffle(indices_list)
        train_ratio, valid_ratio = ratio[0], ratio[1]
        indicies_for_splitting = [int(len(indices_list) * train_ratio), int(len(indices_list) * (train_ratio + valid_ratio))]
        train, val, test = np.split(indices_list, indicies_for_splitting)
        return train, val, test

    def build(self):
        trn_data = copy.copy(self)
        trn_data.data = self._trn
        val_data = copy.copy(self)
        val_data.data = self._val
        tst_data = copy.copy(self)
        tst_data.data = self._tst
        return trn_data, val_data, tst_data
    
    def __len__(self):
        return self.data[0].shape[0]
    
    def __getitem__(self, index):
        batch = {}
        batch['feat'] = torch.from_numpy(
            self.data[0][index].toarray().squeeze()).float()

        batch['target'] = torch.from_numpy(
            self.data[1].indices[self.data[1].indptr[index]: self.data[1].indptr[index + 1]] + 1).long()
        return batch

    def loader(self, batch_size, shuffle, num_workers):
        return DataLoader(self, batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_collate_valid)

    def train_loader(self, batch_size, num_workers):
        return self.loader(batch_size, True, num_workers)
    
    def eval_loader(self, batch_size, num_workers):
        return self.loader(batch_size, False, num_workers)

    
    @property
    def num_items(self):
        return self.num_labels + 1
    
    @property
    def item_freq(self):
        return self.label_freq




if __name__ == '__main__':
    config = {'split_ratio' : [0.7, 0.2, 0.1]}
    dataset = ExtremeClassDataset('mediamill', config)

    trn, val, tst = dataset.build()

    trn_loader = trn.train_loader(23, 0)
    val_loader = val.eval_loader(37, 0)
    tst_loader = tst.eval_loader(19, 0)

    for batch in trn_loader:
        print(batch)
        # break
        import pdb; pdb.set_trace()
    for batch in val_loader:
        print(batch)
        break
    for batch in tst_loader:
        print(batch)
        break
    
    import pdb; pdb.set_trace()