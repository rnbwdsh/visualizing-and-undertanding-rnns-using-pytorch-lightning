import pytorch_lightning as pl
import torch
from torch import sigmoid, tanh
from torch.nn import *
from torch.nn.functional import one_hot

import dataloader
from config import *


# from LSTM import LSTM  # to use huangho lstm


class CharRNN(pl.LightningModule):
    # class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, model_name, dropout, n_layers, lr):
        super(CharRNN, self).__init__()

        name_cell = {"lstm": LSTM, "rnn": RNN, "gru": GRU}
        constructor = name_cell[model_name]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.lr = lr
        self.n_layers = n_layers

        if embedding_dim != 0:
            self.encoder = Embedding(vocab_size, vocab_size)
        else:
            def encoder(x):
                return one_hot(x, num_classes=self.vocab_size).float()

            self.encoder = encoder
        self.cell = constructor(vocab_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.decoder = Linear(hidden_size, vocab_size)  # * 2 if constructor == LSTMCell else 1
        self.dropout = Dropout(dropout)
        self.do_encode = embedding_dim != 0

    def forward(self, x):
        seq_len, batch_size = x.shape
        x = self.encoder(x)
        x = self.cell(x)
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
        self.eval()
        x = self.encoder(x)
        c = self.cell
        hidden = self.hidden_size
        gates, outputs = [], []
        if type(c) == torch.nn.LSTM:
            if c.batch_first:
                x = x.transpose(0, 1)
            # init 0 for start
            hx, cx = [torch.zeros(self.n_layers, x.shape[1], self.hidden_size).to(x.device) for _ in range(2)]
            for xt in x:
                ig, fg, cg, og = 0, 0, 0, 0  # to avoid reference-before-assignment-warning
                for weight_input, weight_hidden, bias_input, bias_hidden in c.all_weights:  # iterate over all layers
                    # documentation says every 4*hidden vector is Wi = W_ii|W_if|W_ic|W_io
                    wii, wif, wic, wio = weight_input.view(4, hidden, -1)
                    whi, whf, whc, who = weight_hidden.view(4, hidden, -1)
                    bii, bif, bic, bio = bias_input.view(4, hidden)
                    bhi, bhf, bhc, bho = bias_hidden.view(4, hidden)

                    ig = (xt @ wii.T + bii) + (cx @ whi.T + bhi)
                    fg = (xt @ wif.T + bif) + (cx @ whf.T + bhf)
                    cg = (xt @ wic.T + bic) + (cx @ whc.T + bhc)
                    og = (xt @ wio.T + bio) + (cx @ who.T + bho)

                    # tmp = (xt @ weight_input.T + bias_input) + (cx @ weight_hidden.T + bias_hidden)
                    # ig, fg, cg, og = [tmp[:, i:i+hidden] for i in range(0, hidden*4, hidden)]
                    ig, fg, cg, og = sigmoid(ig), sigmoid(fg), tanh(cg), sigmoid(og)

                    cx = fg * cx + ig * cg
                    xt = hx = og * tanh(cx)
                outputs.append(hx)
                gates.append([ig, fg, cg, og])

            outputs = torch.cat(outputs)
            if c.batch_first:
                outputs2, (cx2, hx2) = c(x.transpose(0, 1))
                outputs2 = outputs2.transpose(0, 1)
            else:
                outputs2, (cx2, hx2) = c(x)
            print(outputs.sum(), outputs2.sum())
            print(cx.sum(), cx2.sum())
            assert outputs.shape == outputs2.shape
            assert cx.shape == cx2.shape
            assert hx.shape == hx.shape
        self.train()
        return gates

    # pytorch-lightning from here on
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]

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
        self.extract_gates(batch[0], )
        return {'val_loss': loss, "val_acc": acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = sum([x['val_acc'] for x in outputs]) / len(outputs)
        print("avg acc", avg_acc)
        return {'val_loss': avg_loss, 'val_acc': avg_acc}
