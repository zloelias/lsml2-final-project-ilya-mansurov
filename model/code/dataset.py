import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

np.random.seed(42)

class SmotrimDataset(Dataset):

    def __init__(self, file_name):
        self.df = pd.read_csv(file_name, header=0)
        self.df['user_id'] = self.df['user_id'].astype('category')
        self.df['item_id'] = self.df['item_id'].astype('category')
        self.df['user_ind'] = self.df['user_id'].cat.codes
        self.df['item_ind'] = self.df['item_id'].cat.codes
        self.df['rating'] = self.df['rating'].apply(lambda x: 0 if x==-1 else 1)
        self.n_users = len(self.df['user_ind'].unique())
        self.n_items = len(self.df['item_ind'].unique())

        self.data = torch.tensor(self.df[['user_ind', 'item_ind', 'rating']].values).to(torch.int64)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return self.data[index]

