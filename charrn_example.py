# based on Character-To-Character RNN With Pytorch’s LSTMCell by Stepan Ulyanin
# from https://medium.com/coinmonks/character-to-character-rnn-with-pytorchs-lstmcell-cd923a6d0e72
# - comments, + imports, + missing methods
import numpy as np
import torch
from tensorflow.keras.utils import to_categorical

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CHAR = 128
N_SEQ_IN_BATCH = 128
PREDICT_SEQ_LEN = 128

with open('data/warandpeace.txt', 'r') as f:
    text = f.read()
characters = tuple(set(text))
int2char = dict(enumerate(characters))
char2int = {char: index for index, char in int2char.items()}
encoded = np.array([char2int[char] for char in text])


def get_batches(arr, n_seqs_in_a_batch, n_characters):
    batch_size = n_seqs_in_a_batch * n_characters
    n_batches = len(arr) // batch_size
    arr = arr[:n_batches * batch_size]
    arr = arr.reshape((n_seqs_in_a_batch, -1))
    for n in range(0, arr.shape[1], n_characters):
        x = arr[:, n:n + n_characters]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_characters]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


class CharLSTM(torch.nn.ModuleList):
    def __init__(self, sequence_len, vocab_size, hidden_dim, batch_size):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.lstm_1 = torch.nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_dim)
        self.lstm_2 = torch.nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc = torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x, hc):
        output_seq = torch.empty((self.sequence_len, self.batch_size, self.vocab_size), device=x.device)
        hc_1, hc_2 = hc, hc
        for t in range(self.sequence_len):
            h_1, c_1 = self.lstm_1(x[t], hc_1)
            h_2, c_2 = self.lstm_2(h_1, hc_2)
            output_seq[t] = self.fc(self.dropout(h_2))
        return output_seq.view((self.sequence_len * self.batch_size, -1))

    def init_hidden(self):
        return (torch.zeros(self.batch_size, self.hidden_dim).to(DEVICE),
                torch.zeros(self.batch_size, self.hidden_dim).to(DEVICE))

    def init_hidden_predict(self):
        return (torch.zeros(1, self.hidden_dim).to(DEVICE),
                torch.zeros(1, self.hidden_dim).to(DEVICE))

    def predict(self, char, top_k=5, seq_len=128):
        self.eval()
        seq = np.empty(seq_len + 1)
        seq[0] = char2int[char]
        hc = self.init_hidden_predict()
        char = to_categorical(char2int[char], num_classes=self.vocab_size)
        char = torch.from_numpy(char).unsqueeze(0).to(DEVICE)
        for t in range(seq_len):
            h_1, _ = self.lstm_1(char, hc)
            h_2, _ = self.lstm_2(h_1, hc)
            h_2 = self.fc(h_2)
            h_2 = h_2.softmax(dim=1)
            p, top_char = h_2.topk(top_k)
            top_char = top_char.squeeze().cpu().numpy()
            p = p.detach().squeeze().cpu().numpy()
            char = np.random.choice(top_char, p=p / p.sum())
            seq[t + 1] = char
            char = to_categorical(char, num_classes=self.vocab_size)
            char = torch.from_numpy(char).unsqueeze(0).to(DEVICE)
        return seq


net = CharLSTM(sequence_len=N_CHAR, vocab_size=len(char2int), hidden_dim=512, batch_size=128).to(DEVICE)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
val_idx = int(len(encoded) * (1 - 0.1))
data, val_data = encoded[:val_idx], encoded[val_idx:]
val_losses = list()
samples = list()
for epoch in range(10):
    hc = net.init_hidden()
    for i, (x, y) in enumerate(get_batches(data, n_seqs_in_a_batch=N_SEQ_IN_BATCH, n_characters=N_CHAR)):
        x_train = torch.from_numpy(to_categorical(x, num_classes=net.vocab_size).transpose([1, 0, 2])).to(DEVICE)
        targets = torch.from_numpy(y.T).to(DEVICE, dtype=torch.long)
        optimizer.zero_grad()
        output = net(x_train, hc)
        loss = criterion(output, targets.contiguous().view(N_SEQ_IN_BATCH * N_CHAR))
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            val_h, val_c = net.init_hidden()
            val_loss = 0
            for val_x, val_y in get_batches(val_data, n_seqs_in_a_batch=N_SEQ_IN_BATCH, n_characters=N_CHAR):
                val_x = torch.from_numpy(to_categorical(val_x, num_classes=net.vocab_size).transpose([1, 0, 2])).to(
                    DEVICE)
                val_y = torch.from_numpy(val_y.T).to(dtype=torch.long).contiguous().view(
                    N_SEQ_IN_BATCH * N_CHAR).to(DEVICE, dtype=torch.long)
                val_output = net(val_x, (val_h, val_c))
                val_loss = criterion(val_output, val_y)
                val_losses.append(val_loss.item())
                samples.append(''.join([int2char[int_] for int_ in net.predict("A", seq_len=PREDICT_SEQ_LEN)]))
            print("Epoch: {}, Batch: {}, Train Loss: {:.6f}, Validation Loss: {:.6f}".format(epoch, i, loss.item(),
                                                                                             val_loss.item()))
