from utils import *
import json
import os.path as path
import numpy as np
from utils.plot import plot_gate
import sys
import torch
import dataloader
from net import CharRNN
from config import *


def get_saturated(gate, left_thresh, right_thresh):
    left_s = []  # length = num_layers
    right_s = []
    total_seq_length = SEQ_LEN # total_seq_length = total character number
    for i in range(gate.shape[0]):  # for each layer
        left_tmp = gate[i] < left_thresh  # gate[i]: (total_seq_length, hidden_size)
        right_tmp = gate[i] > right_thresh
        left_tmp = torch.sum(left_tmp, dim =0) / total_seq_length  # boradcasting
        right_tmp = torch.sum(right_tmp, dim=0) / total_seq_length
        # add to a list
        left_s.append(left_tmp)
        right_s.append(right_tmp)

    return left_s, right_s  # left_s/right_s: (hidden_size)


def visgate(input_gate, forget_gate, output_gate):
    # visualize gate value
    left_thresh = 0.1  # left saturated threshold
    right_thresh = 0.9  # right saturated threshold

    input_left_s, input_right_s = get_saturated(input_gate, left_thresh, right_thresh)
    forget_left_s, forget_right_s = get_saturated(forget_gate, left_thresh, right_thresh)
    output_left_s, output_right_s = get_saturated(output_gate, left_thresh, right_thresh)
    plot_gate((input_left_s, input_right_s), (forget_left_s, forget_right_s), (output_left_s, output_right_s))


def viscell(cell, vis_dir, max_length=10000):
    # write seq and cell into a json file for visualization
    seq = []
    seq.extend(x[:max_length//BATCH_SIZE])
    char_cell = {}
    char_cell['cell_size'] = HIDDEN_SIZE
    char_cell['seq'] = ''.join(seq)

    # allocate space for cell values
    for j in range(N_LAYERS):
        char_cell['cell_layer_' + str(j + 1)] = []

    total_char = len(cell)
    for i in range(total_char):  # for each character (time step)
        for j in range(N_LAYERS):  # for each layer
            char_cell['cell_layer_' + str(j + 1)].append(cell[i][j])#error here regarding serilization

    with open(path.join(vis_dir, 'char_cell.json'), 'w') as json_file:
        json.dump(char_cell, json_file)


# Vis gate
(trl, tel, val), vocab = dataloader.load(FILE_PATH, DEVICE, SPLITS, BATCH_SIZE, SEQ_LEN, unique=True)
net = CharRNN(len(vocab), HIDDEN_SIZE, EMBEDDING_DIM, MODEL_NAME, DROPOUT, N_LAYERS, DEVICE)
name = f"{MODEL_NAME}-{N_LAYERS}-{HIDDEN_SIZE}"
model = f"models/{name}-warandpeace.pkl"
state_dict = torch.load(model,map_location=torch.device('cpu'))
net.load_state_dict(state_dict)
net.eval()
if MODEL_NAME == 'lstm':
    sequ = []
    input_gates, forget_gates, cell_gates, output_gates = [], [], [], []  # here I will do the most frustrating think you will ever see sorry markus
    cell_states = []
elif MODEL_NAME == 'gru':
    reset_gates, update_gates = [], []
else:
    raise TypeError('This model does not have gates')
for x, y in tel:
    for i in range(N_LAYERS):
        if MODEL_NAME == 'lstm':
            input_gate, forget_gate, cell_gate, output_gate = net.extract_gates(x)[2][i]
            cell_states.append(net.extract_gates(x)[1])
            input_gates.append(input_gate)
            forget_gates.append(forget_gate)
            cell_gates.append(cell_gates)
            output_gates.append(output_gate)
            sequ.extend(x)
        else:
            reset_gate, update_gate = net.extract_gates(x)[2]
            reset_gates.append(reset_gate)
            update_gates.append(update_gate)
if MODEL_NAME == 'lstm':
    #visgate(input_gates[0], forget_gates[0], output_gates[0])
    viscell(cell_states,dataloader.decode(torch.cat(sequ),vocab),'output')
elif MODEL_NAME == 'gru':
    visgate(np.array(reset_gates), np.array(update_gates))
