# strongly based on https://github.com/huanghao-code/VisRNN_ICLR_2016_Text/blob/master/model/LSTM.py
from typing import Union

from torch import tanh, sigmoid, cat, zeros, stack
from torch.nn import LSTM, RNN, GRU


class LSTMx(LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(LSTMx, self).__init__(input_size, hidden_size, batch_first=batch_first, num_layers=num_layers)

    def forward_extract_verify(self, x):
        outputs, (hx, cx), gates = forward_extract(self, x)
        outputs2, (hx2, cx2) = self.forward(x)
        assert outputs.shape == outputs2.shape
        assert hx.shape == cx2.shape
        assert cx.shape == cx.shape
        assert outputs.sum() == outputs2.sum()
        assert cx.sum() == cx2.sum()
        assert hx.sum() == hx2.sum()
        return outputs, (hx, cx), gates

    def recurrence(self, xt, hidden):
        hx, cx = hidden
        xto, yto = [], []
        gates = []
        for i in range(self.num_layers):
            # documentation says every 4*hidden vector is Wi = W_ii|W_if|W_ic|W_io
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


class RNNx(RNN):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(RNNx, self).__init__(input_size, hidden_size, batch_first=batch_first, num_layers=num_layers)

    def forward_extract_verify(self, x):
        outputs, cx, gates = forward_extract(self, x)
        outputs2, cx2 = self.forward(x)
        assert outputs.shape == outputs2.shape
        assert cx.shape == cx2.shape
        return outputs, cx, gates

    def recurrence(self, x, hidden):
        output, hidden = self.forward(x, hidden)
        return hidden, hidden


class GRUx(GRU):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(GRUx, self).__init__(input_size, hidden_size, batch_first=batch_first, num_layers=num_layers)

    def forward_extract_verify(self, x):
        outputs, cx, gates = forward_extract(self, x)
        outputs2, cx2 = self.forward(x)
        assert outputs.shape == outputs2.shape
        assert outputs.sum() == outputs2.sum()
        assert cx.shape == cx2.shape
        return outputs, cx, gates

    def recurrence(self, x, hidden):
        gates = []
        for i in range(self.num_layers):
            # TODO: https://blog.floydhub.com/gru-with-pytorch/ similar to LSTM
            weight_input, weight_hidden, bias_input, bias_hidden = self.all_weights[i]
            gate_reset = sigmoid(x @ weight_input.T + bias_input + hidden[i] @ weight_hidden + bias_hidden)
        return hidden, hidden


def forward_extract(net: Union[LSTMx, RNNx, GRUx], x):  # works like an LSTM forward but additionally outputs gates
    net.eval()
    if net.batch_first:
        x = x.transpose(0, 1)

    outputs = []
    gates = []

    if type(net) == LSTMx:  # for lstms, the hidden is a tuple
        hidden = [zeros(net.num_layers, x.shape[1], net.hidden_size).to(x.device) for _ in range(2)]
    else:
        hidden = zeros(net.num_layers, x.shape[1], net.hidden_size)

    for t in range(len(x)):
        hidden, gate = net.recurrence(x[t:t + 1], hidden)
        if type(hidden) == tuple:  # for output, only the last layer is relevant. for LSTM the hidden is a tuple
            outputs.append(hidden[0][-1])
        else:
            outputs.append(hidden[-1])
        gates.append(gate)
    outputs = stack(outputs)

    if net.batch_first:
        outputs = outputs.transpose(0, 1)
    net.train()
    return outputs, hidden, gates
