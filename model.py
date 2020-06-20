import pytorch_lightning as pl
import torch
from torch.nn import LSTMCell, RNNCell, GRUCell, Dropout, Embedding, Linear
from torch.nn.functional import one_hot


class CharRNN(pl.LightningModule):
    def __init__(self, nr_chars, hidden_size, embedding_dim, model="rnn", dropout=0.0, layers=1, lr=0.01):
        super(CharRNN, self).__init__()
        name_cell = {"lstm": LSTMCell, "rnn": RNNCell, "gru": GRUCell}
        constructor = name_cell[model]

        if embedding_dim != 0:
            self.encoder = Embedding(nr_chars, nr_chars)
        self.cell = []
        for i in range(layers):
            cell = constructor(nr_chars if i == 0 else hidden_size, hidden_size, bias=True)
            self.__setattr__(f"cell{i}", value=cell)  # needs to be a direct prop of self, for correct device
            self.cell.append(cell)
        self.decoder = Linear(hidden_size, nr_chars)
        self.dropout = Dropout(dropout)
        self.encode = embedding_dim != 0
        self.nr_chars = nr_chars

        self.lr = lr
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input: torch.FloatTensor):
        batch_size, seq_len = input.shape
        encoded = self.encoder(input) if self.encode else one_hot(input, num_classes=self.nr_chars).float()
        output_state = torch.empty(seq_len, batch_size, self.nr_chars, device=encoded.device)

        for i in range(0, seq_len):
            out, cell_state = encoded[:, i], None
            for cell in self.cell:
                cell_state = cell(out, cell_state)
                out = cell_state[1] if type(cell_state) == tuple else cell_state
            output_state[i] = self.decoder(self.dropout(out))

        return output_state.view(batch_size * seq_len, -1), None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]

    def _step(self, batch):
        out, _ = self(batch[:, 1:])  # train on everything except last
        targets = batch[:, 1:].flatten()
        loss = self.loss(out, targets.flatten())
        acc = float(out.argmax(1).eq(targets).sum()) / len(out)
        return loss, acc

    def training_step(self, batch, batch_nb):
        loss, acc = self._step(batch)
        tensorboard_logs = {'loss': loss, 'accuracy': acc}
        return {**tensorboard_logs, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        return {'val_loss': loss, 'accuracy': acc}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}
