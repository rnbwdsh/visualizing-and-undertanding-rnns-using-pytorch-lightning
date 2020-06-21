import pytorch_lightning as pl
import torch
from torch.nn import LSTM
from torch.nn import LSTMCell, RNNCell, GRUCell, Dropout, Embedding, Linear
# from LSTM import LSTM
from torch.nn.functional import one_hot

import dataloader
from config import *


class CharRNN(pl.LightningModule):
    # class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, model_name, dropout, n_layers, lr, vs=0):
        super(CharRNN, self).__init__()
        # fix for a bug in pytorch lightning checkpointing
        if type(vocab_size) == dict:
            vocab_size = vs

        name_cell = {"lstm": LSTMCell, "rnn": RNNCell, "gru": GRUCell}
        constructor = name_cell[model_name]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.lr = lr
        self.n_layers = n_layers

        if embedding_dim != 0:
            self.encoder = Embedding(vocab_size, vocab_size)
        self.cell = LSTM(vocab_size, hidden_size, num_layers=n_layers)
        """
        self.cell = []
        for i in range(layers):
            cell = constructor(vocab_size if i == 0 else hidden_size, hidden_size)
            self.__setattr__(f"cell{i}", value=cell)  # needs to be a direct prop of self, for correct device
            self.cell.append(cell)
        """
        self.decoder = Linear(hidden_size, vocab_size)  # * 2 if constructor == LSTMCell else 1
        self.dropout = Dropout(dropout)
        self.do_encode = embedding_dim != 0

    def forward(self, x):
        x = self.encoder(x) if self.do_encode else one_hot(x, num_classes=self.vocab_size).float()
        seq_len, batch_size, nr_chars = x.shape
        """
        output_state = torch.empty(seq_len, batch_size, self.vocab_size, device=x.device)

        for i, xt in enumerate(x):
            out, cell_state = xt, None
            for cell in self.cell:
                cell_state = cell(out, cell_state)
                out = cell_state[0] if type(cell_state) == tuple else cell_state
            output_state[i] = self.decoder(self.dropout(torch.cat(cell_state, dim=1)))
        """
        output = self.cell(x)
        output_state = self.decoder(self.dropout(output[0]))

        return output_state.view(batch_size * seq_len, -1)

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

    # pytorch-lightning from here on
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]

    def step(self, batch, acc=True):
        x, y = batch
        out = self.forward(x)  # train on everything except last
        loss = torch.nn.functional.cross_entropy(out, y.flatten())
        if acc:
            acc = float(out.argmax(1).eq(y.flatten()).sum()) / len(out)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, acc=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        loss, acc = self.step(batch)
        print("val acc", acc)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}
