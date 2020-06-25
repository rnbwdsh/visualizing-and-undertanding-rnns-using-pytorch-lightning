# based on https://github.com/huanghao-code/VisRNN_ICLR_2016_Text/blob/master/vis/visgate.py + /utils/plot.py
import json
import os.path as path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from config import *


def _plot_gate(gates):
    num_gates, num_layers, _, _ = gates.shape
    color_base = ["red", "green", "blue", "yellow"]
    gate_names = {2: ["Update", "Reset"], 3: ["Input", "Forget", "Output"]}
    gate_names = gate_names[num_gates]

    fig = plt.figure()
    fig.suptitle('Saturation Plot', fontsize=16)
    for gate_id, gate in enumerate(gates):
        # plot forget gate -------------------------------------------
        ax = fig.add_subplot(1, num_gates, gate_id + 1, aspect='equal', title=f'{gate_names[gate_id]} Gate',
                             xlabel='fraction right saturated', ylabel='fraction left saturated')
        for layer, (right, left) in enumerate(gate):
            # scatterplot with bigger points for more common points
            clr = Counter(zip(left, right))
            l, r = np.array(list(clr.keys())).T
            rad = 100 * np.array(list(clr.values())) / max(clr.values())
            c = np.full(len(l), color_base[layer])
            plt.scatter(l, r, rad, c, alpha=1.0 / num_layers)
        ax.plot(np.linspace(0, 1), np.linspace(1, 0))  # diagonal
        plt.draw()  # update plot
    plt.show()


def _get_saturation(gate, left_thresh, right_thresh):
    # takes: [layer, textlength, hiddensize), returns [2, layer, textlength]
    return (gate < left_thresh).mean(axis=-1), (gate > right_thresh).mean(axis=-1)


def visualize_gate(*gates):
    gates_lr_layer_text = np.array([_get_saturation(gate, 0.1, 0.9) for gate in gates])
    # reshape to [gate, layer, leftright, data]
    gate_layer_lr_text = gates_lr_layer_text.transpose((0, 2, 1, 3))
    _plot_gate(gate_layer_lr_text)


def visualize_cell(cell, x, vis_dir="visualization"):
    char_cell = {'cell_size': HIDDEN_SIZE, 'seq': ''.join(x)}
    char_cell.update({f"cell_layer_{layer + 1}": cell[layer].tolist() for layer in range(len(cell))})
    with open(path.join(vis_dir, 'char_cell.json'), 'w') as json_file:
        json.dump(char_cell, json_file)
