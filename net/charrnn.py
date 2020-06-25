import numpy as np
import torch
from torch.nn import Linear, Dropout, Embedding, Module

import dataloader
from config import DEVICE
from .grux import GRUx
from .lstmx import LSTMx
from .rnnx import RNNx


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
        self.to(self.device)

    def encode(self, x):
        if self.embedding_dim == 0:
            return torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        else:
            return self.encoder(x)

    def forward(self, x):
        seq_len, batch_size = x.shape
        x = self.encode(x)
        x = self.recurrent(x)
        x = self.dropout(x[0])
        x = self.decoder(x)
        return x.view(batch_size * seq_len, -1)

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

    def extract_gates(self, x):
        x = self.encode(x)
        return self.recurrent.forward_extract(x)[2]

    def extract_from_loader(self, loader):
        epochs = []
        for x, _ in loader:  # run training loop
            x = self.extract_gates(x)
            # 1. reshape so batch_dim is first
            x = np.array(x).transpose((0, 3, 1, 2, 4))
            # 2. concatenate sequences along batch dim
            x = np.concatenate(x)
            # 3. concatenate along epoch dimension
            epochs.append(x)
        # 4. set gate_id as dimension 0 -> unbox to 4 variables with [layer, textpos, gate]
        return np.concatenate(epochs).transpose((2, 1, 0, 3))

    def name(self):
        return f"{self.model_name}-{self.n_layers}-{self.hidden_size}"

    def save_to_file(self):
        self._forward_hooks.clear()  # must be empty to be pickle-able
        with open(f"models/{self.name()}.pkl", "wb") as f:
            torch.save(self, f)

    @staticmethod
    def load_from_file(model_name, nr_layers, hidden_size):
        with open(f"models/{model_name}-{nr_layers}-{hidden_size}.pkl", "rb") as f:
            n = torch.load(f)
            net = CharRNN(n.vocab_size, n.hidden_size, n.embedding_dim, n.model_name, n.dropout.p, n.n_layers, DEVICE)
            net.load_state_dict(n.state_dict())
            return net
