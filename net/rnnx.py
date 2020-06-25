from extractable import Extractor
from torch import tanh, cat
from torch.nn import RNN


class RNNx(RNN, Extractor):
    def recurrence(self, xt, hidden):
        gates = []
        hidden = [*hidden]  # unpack tensor into list
        for t in range(self.num_layers):
            wi, wh, bi, bh = self.all_weights[t]
            xt = tanh(xt @ wi.T + bi + hidden[t] @ wh.T + bh)
            hidden.append(xt)
            gates.append(xt.squeeze().tolist())
        return cat(hidden[self.num_layers:]), [gates]
