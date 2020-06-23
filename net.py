import pytorch_lightning as pl
import torch
from torch.nn import Linear, Dropout, Embedding, Module

import dataloader
from extractable import LSTMx, RNNx, GRUx


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


class CharRNN(Module):
    # class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, model_name, dropout, n_layers, device):
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.n_layers = n_layers
        self.device = device
        self.decoder = Linear(hidden_size, vocab_size)  # * 2 if constructor == LSTMCell else 1
        self.dropout = Dropout(dropout)
        self.do_encode = embedding_dim != 0

        name_cell = {"lstm": LSTMx, "rnn": RNNx, "gru": GRUx}
        constructor = name_cell[model_name]
        self.recurrent = constructor(vocab_size, hidden_size, num_layers=n_layers, batch_first=True)
        if embedding_dim != 0:
            self.encoder = Embedding(vocab_size, vocab_size)
        else:
            self.encoder = lambda x: torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        self.to(self.device)

    def forward(self, x):
        seq_len, batch_size = x.shape
        x = self.encoder(x)
        x = self.recurrent(x)
        x = self.dropout(x[0])
        x = self.decoder(x)
        x = x.view(batch_size * seq_len, -1)
        return x

    def predict(self, x, seq_len, vocab, method="rand"):
        self.eval()
        buffer = torch.zeros(1, len(x) + seq_len, dtype=torch.long, device=self.device)
        offset = len(x)
        buffer[0, :offset] = dataloader.encode(x, vocab, device=self.device)
        for i in range(offset, offset + seq_len):
            out = self.forward(buffer[:, :i])[-1]
            if method == "max":
                out = out.argmax()
            elif method == "rand":
                out ** 2
                out = (out / out.sum()).cumsum(dim=0)
            elif method == "softrand":
                out = out.softmax(dim=0).cumsum(dim=0)
            if "rand" in method:
                out = (out < torch.rand(1, device=out.device)).sum()
            buffer[:, i] = out
            # print(*[k for k, v in vocab.items() if v == out], end="")
        decoded = dataloader.decode(buffer[0], vocab)
        self.train()
        return decoded

    def extract_gates(self, x):
        x = self.encoder(x)
        return self.recurrent.forward_extract(x)[2]
