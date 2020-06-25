import pytorch_lightning as pl
import torch


class Lightning(pl.LightningModule):
    def __init__(self, net, lr):
        super(Lightning, self).__init__()
        self.net = net
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.95)

    def forward(self, x):
        return self.net.forward(x)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def step(self, batch):
        x, y = batch
        out = self.forward(x)  # train on everything except last
        loss = torch.nn.functional.cross_entropy(out, y.flatten())
        acc = float(out.argmax(1).eq(y.flatten()).sum()) / len(out)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        return {'loss': loss, 'progress_bar': {'acc': acc}}

    def validation_step(self, batch, batch_nb):
        loss, acc = self.step(batch)
        return {'val_loss': loss, "val_acc": acc}

    def test_step(self, batch, batch_nb):
        loss, acc = self.step(batch)
        return {'test_loss': loss, "test_acc": acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = sum([x['val_acc'] for x in outputs]) / len(outputs)
        return {'val_loss': avg_loss, 'val_acc': avg_acc}
