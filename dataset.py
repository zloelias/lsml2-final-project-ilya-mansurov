import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset

np.random.seed(42)


class SmotrimDataset(Dataset):

    def __init__(self, file_name):
        df = pd.read_csv(file_name, header=None, names=['user_id', 'item_id', 'rating'])


        user_c = CategoricalDtype(sorted(df['user_id'].unique()), ordered=True)
        item_c = CategoricalDtype(sorted(df['item_id'].unique()), ordered=True)

        row = df['user_id'].astype(person_c).cat.codes
        col = df['item_id'].astype(thing_c).cat.codes
        self.matrix = csr_matrix(
            (df['rating'], (row, col)),
            shape=(person_c.categories.size, thing_c.categories.size)
        )

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, index):
        return self.matrix.getrow(index)
