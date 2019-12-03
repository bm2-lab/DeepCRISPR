import pandas as pd
import numpy as np
from operator import add
from functools import reduce

__all__ = ['Sgt', 'Episgt', 'Epiotrt']

ntmap = {'A': (1, 0, 0, 0),
         'C': (0, 1, 0, 0),
         'G': (0, 0, 1, 0),
         'T': (0, 0, 0, 1)
         }
epimap = {'A': 1, 'N': 0}


def get_seqcode(seq):
    return np.array(reduce(add, map(lambda c: ntmap[c], seq.upper()))).reshape(
        (1, len(seq), -1))


def get_epicode(eseq):
    return np.array(list(map(lambda c: epimap[c], eseq))).reshape(1, len(eseq), -1)


class Sgt:
    def __init__(self, fpath, with_y=True):
        self._fpath = fpath
        self._ori_df = pd.read_csv(fpath, sep='\t', index_col=None, header=None)
        self._with_y = with_y
        self._num_cols = 2 if with_y else 1
        self._cols = list(self._ori_df.columns)[-self._num_cols:]
        self._df = self._ori_df[self._cols]

    @property
    def length(self):
        return len(self._df)

    def get_dataset(self, x_dtype=np.float32, y_dtype=np.float32):
        x_seq = np.concatenate(list(map(get_seqcode, self._df[self._cols[0]])))
        x = x_seq.astype(dtype=x_dtype)
        x = x.transpose(0, 2, 1)
        if self._with_y:
            y = np.array(self._df[self._cols[-1]]).astype(y_dtype)
            return x, y
        else:
            return x


class Episgt:
    def __init__(self, fpath, num_epi_features, with_y=True):
        self._fpath = fpath
        self._ori_df = pd.read_csv(fpath, sep='\t', index_col=None, header=None)
        self._num_epi_features = num_epi_features
        self._with_y = with_y
        self._num_cols = num_epi_features + 2 if with_y else num_epi_features + 1
        self._cols = list(self._ori_df.columns)[-self._num_cols:]
        self._df = self._ori_df[self._cols]

    @property
    def length(self):
        return len(self._df)

    def get_dataset(self, x_dtype=np.float32, y_dtype=np.float32):
        x_seq = np.concatenate(list(map(get_seqcode, self._df[self._cols[0]])))
        x_epis = np.concatenate([np.concatenate(list(map(get_epicode, self._df[col]))) for col in
                                 self._cols[1: 1 + self._num_epi_features]], axis=-1)
        x = np.concatenate([x_seq, x_epis], axis=-1).astype(x_dtype)
        x = x.transpose(0, 2, 1)
        if self._with_y:
            y = np.array(self._df[self._cols[-1]]).astype(y_dtype)
            return x, y
        else:
            return x

class Epiotrt:
    def __init__(self, fpath, num_epi_features, with_y=True):
        self._fpath = fpath
        self._ori_df = pd.read_csv(fpath, sep='\t', index_col=None, header=None)
        self._num_epi_features = num_epi_features
        self._with_y = with_y
        self._num_cols = num_epi_features * 2 + 3 if with_y else num_epi_features * 2 + 2
        self._cols = list(self._ori_df.columns)[-self._num_cols:]
        self._df = self._ori_df[self._cols]

    @property
    def length(self):
        return len(self._df)

    def get_dataset(self, x_dtype=np.float32, y_dtype=np.float32):
        x_on_seq = np.concatenate(list(map(get_seqcode, self._df[self._cols[0]])))

        x_on_epis = np.concatenate([np.concatenate(list(map(get_epicode, self._df[col]))) for col in
                                 self._cols[1: 1 + self._num_epi_features]], axis=-1)
        x_on = np.concatenate([x_on_seq, x_on_epis], axis=-1).astype(x_dtype)
        x_on = x_on.transpose(0, 2, 1)
        x_off_seq = np.concatenate(list(map(get_seqcode, self._df[self._cols[1 + self._num_epi_features]])))

        x_off_epis = np.concatenate([np.concatenate(list(map(get_epicode, self._df[col]))) for col in
                                 self._cols[2 + self._num_epi_features: 2 + self._num_epi_features * 2]], axis=-1)
        x_off = np.concatenate([x_off_seq, x_off_seq], axis=-1).astype(x_dtype)
        x_off = x_off.transpose(0, 2, 1)


        if self._with_y:
            y = np.array(self._df[self._cols[-1]]).astype(y_dtype)
            return (x_on, x_off), y
        else:
            return (x_on, x_off)

