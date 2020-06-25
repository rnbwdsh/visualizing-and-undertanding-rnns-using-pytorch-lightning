from torch import cat, sigmoid, tanh
from torch.nn import GRU

from .extractable import Extractor


class GRUx(GRU, Extractor):
    def recurrence(self, xt, hidden):
        gates = []
        hidden = [*hidden]
        for t in range(self.num_layers):
            # https://pytorch.org/docs/master/generated/torch.nn.GRU.html
            weight_input, weight_hidden, bias_input, bias_hidden = self.all_weights[t]

            wir, wiz, win = weight_input.view(3, self.hidden_size, -1)
            whr, whz, whn = weight_hidden.view(3, self.hidden_size, -1)
            bir, biz, bin = bias_input.view(3, self.hidden_size)
            bhr, bhz, bhn = bias_hidden.view(3, self.hidden_size)

            r = sigmoid(xt @ wir.T + bir + hidden[t] @ whr.T + bhr.T)
            z = sigmoid(xt @ wiz.T + biz + hidden[t] @ whz.T + bhz.T)
            n = tanh(xt @ win.T + bin + r * (hidden[t] @ whn.T + bhn))
            xt = (1 - z) * n + z * hidden[t]
            hidden.append(xt)
            gates.append([r.squeeze(dim=0).tolist(), z.squeeze(dim=0).tolist()])
        return cat(hidden[self.num_layers:]), gates
