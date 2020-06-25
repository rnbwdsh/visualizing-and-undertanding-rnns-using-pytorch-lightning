# strongly based on https://github.com/huanghao-code/VisRNN_ICLR_2016_Text/blob/master/model/LSTM.py
from torch import tanh, sigmoid, cat
from torch.nn import LSTM

from .extractable import Extractor


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

            gates.append([g.squeeze(dim=0).tolist() for g in
                          [input_gate, forget_gate, cell_gate, output_gate, cy]])
            xto.append(xt)
            yto.append(cy)
        return (cat(xto), cat(yto)), gates
