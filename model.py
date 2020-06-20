import pytorch_lightning as pl
import torch
from torch.nn import LSTMCell, RNNCell, GRUCell, Dropout, Embedding, Linear, CrossEntropyLoss
from torch.nn.functional import one_hot


class CharRNN(pl.LightningModule):
    def __init__(self, nr_chars, hidden_size, embedding_dim, model="rnn", lr=0.01, dropout=0.0):
        super(CharRNN, self).__init__()
        name_cell = {"lstm": LSTMCell, "rnn": RNNCell, "gru": GRUCell}
        constructor = name_cell[model]

        if embedding_dim != 0:
            self.encoder = Embedding(embedding_dim, nr_chars)
        self.cell = constructor(nr_chars, hidden_size, bias=True)
        self.decoder = Linear(hidden_size, nr_chars)
        self.dropout = Dropout(dropout)
        self.encode = embedding_dim != 0
        self.nr_chars = nr_chars

        self.loss = CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, input: torch.FloatTensor):
        batch_size, seq_len = input.shape
        if self.encode:
            encoded = self.encoder(input.T).transpose(0, 1).contiguous()  # contiguous alligns it in memory
        else:
            encoded = one_hot(input, num_classes=self.nr_chars).float()
        cs = [self.cell(encoded[:, 0])]
        for i in range(1, seq_len):  # skip first element
            cs.append(self.cell(encoded[:, i], cs[-1]))

        if isinstance(cs[0], tuple):
            decoded = self.decoder(torch.cat([c[0] for c in cs]))
            decoder2 = [c[1] for c in cs]
        else:
            decoded = self.decoder(torch.cat(cs))
        # decoded = self.decoder(cs[-1])  # test code: only return last
        return decoded.softmax(-1)

    def _step(self, batch):
        out = self(batch[:, :-1])  # train on everything except last
        targets = batch[:, 1:].flatten()
        loss = self.loss(out, targets)
        acc = float((out.argmax(1) == targets).sum()) / len(out)
        return loss, acc

    def training_step(self, batch, batch_nb):
        loss, acc = self._step(batch)
        tensorboard_logs = {'loss': loss, 'accuracy': acc}
        return {**tensorboard_logs, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return self.optimizer

    """
    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        return {'val_loss': loss, 'accuracy': acc}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}
"""