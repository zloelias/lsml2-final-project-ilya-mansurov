import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score


class MLP(pl.LightningModule):

    def __init__(self, n_users, n_items, layers=[16, 8], dropout=False, lr=1e-3):
        super().__init__()
        self._target_ratings = []
        self._pred_ratings = []
        self.lr = lr
        assert (layers[0] % 2 == 0), "layers[0] must be an even number"
        self.__alias__ = "MLP {}".format(layers)
        self.__dropout__ = dropout

        # user and item embedding layers
        embedding_dim = int(layers[0]/2)
        self.user_embedding = torch.nn.Embedding(n_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(n_items, embedding_dim)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, x):
        users, items = x[:,0], x[:,1]
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        # concatenate user and item embeddings to form input
        x = torch.cat([user_embedding, item_embedding], 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x,  p=self.__dropout__, training=self.training)
        logit = self.output_layer(x)
        rating = torch.sigmoid(logit)
        return rating

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss(self, pred, target):
        loss = nn.BCELoss()
        return loss(pred, target)

    def training_step(self, batch, batch_idx):
        x, ratings = batch[:,:2], batch[:,2]
        ratings = ratings.float()
        pred_ratings = self.forward(x)
        loss = self.loss(pred_ratings.reshape(-1, 1), ratings.reshape(-1, 1))
        self.log(f"train_loss", loss, on_epoch=True)
        self.log(f"train_loss__{self.current_epoch}", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, ratings = batch[:,:2], batch[:,2]
        ratings = ratings.float()
        pred_ratings = self.forward(x)
        self._target_ratings += ratings.reshape(-1).tolist()
        self._pred_ratings += pred_ratings.reshape(-1).tolist()
        loss = self.loss(pred_ratings.reshape(-1, 1), ratings.reshape(-1, 1))
        self.log(f"val_loss", loss, on_epoch=True)
        self.log(f"val_loss_{self.current_epoch}", loss, on_epoch=True)
        return loss

    def validation_epoch_end(self, processed_epoch_output):
        acc = roc_auc_score(self._target_ratings, self._pred_ratings)
        self.log(f"val_roc_auc_{self.current_epoch}", acc)
        self._target_ratings = []
        self._pred_ratings = []

    def get_alias(self):
        return self.__alias__


