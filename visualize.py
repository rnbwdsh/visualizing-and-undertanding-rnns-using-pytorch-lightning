import json
import os.path as path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

import dataloader
from config import *
from net import CharRNN


def plot_gate(input_gate, forget_gate, output_gate):
    # prepare data
    (input_left_s, input_right_s), (forget_left_s, forget_right_s), (output_left_s, output_right_s) = \
        input_gate, forget_gate, output_gate
    # num of layers
    num_layers = len(input_gate)

    fig = plt.figure()

    # super title
    fig.suptitle('Saturation Plot', fontsize=16)

    color_base = [100, 60, 30]

    # plot forget gate -------------------------------------------
    ax = fig.add_subplot(1, 3, 2, aspect='equal')
    ax.set_title('Forget Gate')
    ax.set_xlabel('fraction right saturated')
    ax.set_ylabel('fraction left saturated')

    for i in range(num_layers):
        x, y = forget_right_s[i], forget_left_s[i]
        radius = 0.03 * np.ones(len(x))
        patches = []
        for xc, yc, r in zip(x, y, radius):
            circle = Circle((xc, yc), r)
            patches.append(circle)

        colors = color_base[i] * np.ones(len(patches))  # color of circle
        p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1)
        p.set_array(np.array(colors))
        ax.add_collection(p)

    # plot reverse diagnoal
    ax.plot(np.linspace(0, 1), np.linspace(1, 0))

    cb = fig.colorbar(p, ax=ax)
    cb.remove()  # remove color bar
    plt.draw()  # update plot

    # plot input gate -------------------------------------------
    ax = fig.add_subplot(1, 3, 1, aspect='equal')
    ax.set_title('Input Gate')
    ax.set_xlabel('fraction right saturated')
    ax.set_ylabel('fraction left saturated')

    for i in range(num_layers):
        x, y = input_right_s[i], input_left_s[i]
        radius = 0.03 * np.ones(len(x))
        patches = []
        for xc, yc, r in zip(x, y, radius):
            circle = Circle((xc, yc), r)
            patches.append(circle)

        colors = 100 * np.ones(len(patches))  # color of circle
        p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1)
        p.set_array(np.array(colors))
        ax.add_collection(p)

    # plot reverse diagnoal
    ax.plot(np.linspace(0, 1), np.linspace(1, 0))

    cb = fig.colorbar(p, ax=ax)
    cb.remove()  # remove color bar
    plt.draw()  # update plot

    # plot output gate -------------------------------------------
    ax = fig.add_subplot(1, 3, 3, aspect='equal')
    ax.set_title('Output Gate')
    ax.set_xlabel('fraction right saturated')
    ax.set_ylabel('fraction left saturated')

    for i in range(num_layers):
        x, y = output_right_s[i], output_left_s[i]
        radius = 0.03 * np.ones(len(x))
        patches = []
        for xc, yc, r in zip(x, y, radius):
            circle = Circle((xc, yc), r)
            patches.append(circle)

        colors = 100 * np.ones(len(patches))  # color of circle
        p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1)
        p.set_array(np.array(colors))
        ax.add_collection(p)

    # plot reverse diagnoal
    ax.plot(np.linspace(0, 1), np.linspace(1, 0))

    cb = fig.colorbar(p, ax=ax)
    cb.remove()  # remove color bar
    plt.draw()  # update plot

    plt.show()


def get_saturated(gate, left_thresh, right_thresh):
    left_s = []  # length = num_layers
    right_s = []
    for i in range(N_LAYERS):  # for each layer
        left_tmp = gate[i][0] < left_thresh  # gate[i]: (total_seq_length, hidden_size)
        right_tmp = gate[i][0] > right_thresh
        left_tmp = np.sum(left_tmp, 0) / SEQ_LEN  # boradcasting
        right_tmp = np.sum(right_tmp, 0) / SEQ_LEN
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


def viscell(cell, x, vis_dir):
    # write seq and cell into a json file for visualization
    seq = x[:]
    char_cell = {'cell_size': HIDDEN_SIZE, 'seq': ''.join(seq)}
    # allocate space for cell values
    for j in range(N_LAYERS):
        char_cell['cell_layer_' + str(j + 1)] = []

    total_char = len(cell)
    for i in range(total_char):  # for each character (time step)
        for j in range(N_LAYERS):  # for each layer
            char_cell['cell_layer_' + str(j + 1)].append(cell[i][j])  # error here regarding serilization

    with open(path.join(vis_dir, 'char_cell.json'), 'w') as json_file:
        json.dump(char_cell, json_file)


# Vis gate
(trl, tel, val), vocab = dataloader.load(FILE_PATH, DEVICE, SPLITS, BATCH_SIZE, SEQ_LEN, unique=True)
net = CharRNN.load_from_file(MODEL_NAME, N_LAYERS, HIDDEN_SIZE).eval()

c_states, i_gates, f_gates, c_gates, o_gates = [], [], [], [], []
u_gates, r_gates = [], []
# shape: [epoch, seq, layer, gate_id, batch, gate_data]
x, y = next(iter(tel))  # only do one batch
extracted = np.array(net.extract_gates(x))
# reshape to [batch, seq, layer, gate_id, gate_data]
extracted = extracted.transpose(0, 3, 1, 2, 4)
# concatenate sequences along batch
extracted = np.concatenate(extracted)
# set gate_id as dimension 0 -> unbox to 4 variables with [layer, textpos, gate]
extracted = extracted.transpose(2, 1, 0, 3)

if MODEL_NAME == 'lstm':
    input_gates, forget_gates, cell_gates, output_gates, cell_states = extracted
    visgate(input_gates, forget_gates, output_gates)
    viscell(cell_states, dataloader.decode(y, vocab), 'output')
elif MODEL_NAME == "gru":
    reset_gates, update_gates = extracted
    visgate((reset_gates), (update_gates))
