import torch.nn as nn
import pytorch_lightning as pl

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim: int, h_dim: int = 128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(256, h_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Flatten(start_dim=1)
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (h_dim, 1)),
            nn.Linear(h_dim, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(512, input_dim),
            nn.Tanhshrink(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return h

    def decode(self, h):
        z = self.decoder(h)
        return z

    def forward(self, x):
        h = self.encode(x)
        z = self.decode(h)
        return z

    def step(self, x):
        h = self.encode(x)
        z = self.decode(h)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss(self, recon_x, x):
        mse = nn.MSELoss()
        loss = mse(recon_x, x)
        logs = {}
        return loss, logs

    def training_step(self, batch, batch_idx):
        x = batch
        x_recon = self.step(x)
        loss, logs = self.loss(x_recon, x)
        #self.log_dict({f"train_{k}_{self.current_epoch}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, processed_epoch_output):
        # count and log ndcg
        pass

    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon = self.step(x)
        loss, logs = self.loss(x_recon, x)
        #self.log_dict({f"train_{k}_{self.current_epoch}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        x_recon = self.step(x)
        loss, logs = self.loss(x_recon, x)
        #self.log_dict({f"train_{k}_{self.current_epoch}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def test_epoch_end(self, output):
        # count and log ndcg
        pass

    def predict(self, x):
        pass

