from torch import cat, zeros, stack
from torch.nn import LSTM


class Extractor:
    def forward_extract(self, x, verify=True):  # works like an LSTM forward but additionally outputs gates
        x_orig = x  # for later forward pass
        self.eval()
        if self.batch_first:
            x = x_orig.transpose(0, 1)

        if isinstance(self, LSTM):  # for lstms, the hidden is a tuple
            hidden = [zeros(self.num_layers, 1, self.hidden_size, device=x.device) for _ in range(2)]
        else:
            hidden = zeros(self.num_layers, 1, self.hidden_size, device=x.device)

        outputs = []
        gates = []
        for t in range(len(x)):
            hidden, gate = self.recurrence(x[t:t + 1], hidden)
            outputs.append(hidden[0][-1] if type(hidden) == tuple else hidden[-1])
            gates.append(gate)
        outputs = stack(outputs)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        self.train()

        if verify:
            outputs2, hidden2 = self.forward(x_orig)
            assert (outputs - outputs2).mean() < 0.0001
            assert (cat(hidden) - cat(hidden2) if isinstance(self, LSTM) else hidden - hidden2).mean() < 0.0001

        return outputs, hidden, gates
