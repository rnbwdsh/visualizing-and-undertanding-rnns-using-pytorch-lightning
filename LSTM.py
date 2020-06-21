# strongly based on
from torch import tanh, sigmoid, cat, zeros
from torch.nn import ModuleList, Linear, Module


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # note that Linear contains bias
        self.Wii_b, self.Wif_b, self.Wic_b, self.Wio_b = [ModuleList(  # input weight for input gate
            [Linear(input_size, hidden_size)] +
            [Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]) for _ in range(4)]

        self.Whi_b, self.Whf_b, self.Whc_b, self.Who_b = [ModuleList(  # hidden weight for input gate
            [Linear(hidden_size, hidden_size) for _ in range(num_layers)]) for _ in range(4)]

    def forward(self, inputs):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        hx, cx = [zeros(self.num_layers, inputs.shape[1], self.hidden_size).to(inputs.device) for _ in range(2)]
        gates = {}  # a dictionary to receive gate values
        output = []  # a list; call cat() before returned

        for input in inputs:  # process for each time step
            # store four gate values in all layers for t = seq_len (last time step)
            all_input_gate, all_forget_gate, all_cell_gate, all_output_gate = [], [], [], []

            # for each layer
            for layer in range(self.num_layers):
                input_gate = sigmoid(self.Wii_b[layer](input) + self.Whi_b[layer](hx))
                forget_gate = sigmoid(self.Wif_b[layer](input) + self.Whf_b[layer](hx))
                cell_gate = tanh(self.Wic_b[layer](input) + self.Whc_b[layer](hx))
                output_gate = sigmoid(self.Wio_b[layer](input) + self.Who_b[layer](hx))

                # current states
                cx = forget_gate * cx + input_gate * cell_gate
                input = hx = output_gate * tanh(cx)

                # store four gate values in current layer
                all_input_gate.append(input_gate)
                all_forget_gate.append(forget_gate)
                all_cell_gate.append(cell_gate)
                all_output_gate.append(output_gate)

            gates = {'input_gate': all_input_gate, 'forget_gate': all_forget_gate,
                     'cell_gate': all_cell_gate, 'output_gate': all_output_gate}

            output.append(hx[-1].clone().unsqueeze(0))

        output = cat(output, 0)
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (hx, cx), gates  # hidden: (h_n, c_n)
