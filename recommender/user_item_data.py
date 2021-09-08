import pandas as pd
import numpy as np

class userItemData:
    def __init__(self, filename):
        self.df = pd.read_csv(filename, header=0)
        self.df['user_id'] = self.df['user_id'].astype('category')
        self.df['item_id'] = self.df['item_id'].astype('category')
        self.df['user_ind'] = self.df['user_id'].cat.codes
        self.df['item_ind'] = self.df['item_id'].cat.codes

    def get_user_ind_by_user_id(self, user_id):
        res = self.df[self.df['user_id'] == user_id]['user_ind'].unique()
        if len(res) == 0:
            return None
        return res[0]

    def get_not_rated_items_ind_by_user_ind(self, user_ind):
        items = self.df[
            ~self.df['item_ind'].isin(
                self.df[self.df['user_ind'] == user_ind]['item_ind']
            )
        ]['item_ind'].unique()
        return np.array(list(zip([user_ind] * len(items), items)))

    def get_items_id_by_items_ind(self, items_ind):
        return self.df[self.df.item_ind.isin(items_ind)]['item_id'].unique()
