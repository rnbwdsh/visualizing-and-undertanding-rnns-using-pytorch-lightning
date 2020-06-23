# strongly based on https://github.com/huanghao-code/VisRNN_ICLR_2016_Text/blob/master/model/LSTM.py

from torch import tanh, sigmoid, cat, zeros, stack
from torch.nn import LSTM, RNN, GRU


class Extractor:
    def forward_extract(self, x, verify=True):  # works like an LSTM forward but additionally outputs gates
        x_orig = x  # for later forward pass
        self.eval()
        if self.batch_first:
            x = x_orig.transpose(0, 1)

        if type(self) == LSTMx:  # for lstms, the hidden is a tuple
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
            assert (cat(hidden) - cat(hidden2) if type(self) == LSTMx else hidden - hidden2).mean() < 0.0001

        return outputs, hidden, gates


class RNNx(RNN, Extractor):
    def recurrence(self, xt, hidden):
        gates = []
        hidden = [*hidden]  # unpack tensor into list
        for t in range(self.num_layers):
            wi, wh, bi, bh = self.all_weights[t]
            xt = tanh(xt @ wi.T + bi + hidden[t] @ wh.T + bh)
            hidden.append(xt)
            gates.append([xt])
        return cat(hidden[self.num_layers:]), gates


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
            gates.append([r, z])
        return cat(hidden[self.num_layers:]), gates


class LSTMx(LSTM, Extractor):
    def recurrence(self, xt, hidden):
        hx, cx = hidden
        xto, yto, gates = [], [], []
        for i in range(self.num_layers):
            # https://pytorch.org/docs/master/generated/torch.nn.LSTM.html, Wi = W_ii|W_if|W_ic|W_io
            weight_input, weight_hidden, bias_input, bias_hidden = self.all_weights[i]
            wii, wif, wic, wio = weight_input.view(4, self.hidden_size, -1)
            whi, whf, whc, who = weight_hidden.view(4, self.hidden_size, -1)
            bii, bif, bic, bio = bias_input.view(4, self.hidden_size)
            bhi, bhf, bhc, bho = bias_hidden.view(4, self.hidden_size)

            input_gate = sigmoid((xt @ wii.T + bii) + (hx[i] @ whi.T + bhi))
            forget_gate = sigmoid((xt @ wif.T + bif) + (hx[i] @ whf.T + bhf))
            cell_gate = tanh((xt @ wic.T + bic) + (hx[i] @ whc.T + bhc))
            output_gate = sigmoid((xt @ wio.T + bio) + (hx[i] @ who.T + bho))

            cy = forget_gate * cx[i] + input_gate * cell_gate
            xt = output_gate * tanh(cy)

            gates.append([input_gate, forget_gate, cell_gate, output_gate])
            xto.append(xt)
            yto.append(cy)
        return (cat(xto), cat(yto)), gates


