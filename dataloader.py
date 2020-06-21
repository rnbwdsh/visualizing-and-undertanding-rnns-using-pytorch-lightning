from string import printable

import torch
from torch.utils.data import DataLoader, Dataset


class ContStrDataSet(Dataset):
    def __init__(self, data, offset, end, seq_len, unique):
        super(ContStrDataSet, self).__init__()
        self.data = data[offset:end]
        self.sample_len = seq_len + 1
        self.disjoint_samples = unique
        self.len = len(self.data) // self.sample_len if unique else len(self.data) - seq_len

    def __len__(self):  # joined example: end=100, offset=10, sample_len=10 -> starting from 10 to 90
        return self.len

    def __getitem__(self, idx):
        start = idx * self.sample_len if self.disjoint_samples else idx
        batch = self.data[start: start + self.sample_len]
        return batch[:-1], batch[1:]


def load(file_name, device, splits=(0, 80, 90, 100), batch_size=100, seq_len=100, unique=True):
    data = open(file_name).read()
    assert set(data).issubset(printable), "Datset contains non-strings.printable-chars"
    printable_id = {s: i for i, s in enumerate(sorted(set(data)))}
    t = encode(data, printable_id, device)
    splits = [len(data) * i // 100 for i in splits]
    loaders = [DataLoader(ContStrDataSet(t, splits[i], splits[i + 1], seq_len, unique=unique),
                          batch_size=batch_size, shuffle=not bool(splits[i])) for i in range(3)]
    return loaders, printable_id


def encode(data, printable_id, device):
    return torch.tensor([printable_id[c] for c in data], dtype=torch.long, device=device)


def decode(tensor, printable_id):
    id_printable = {v: k for k, v in printable_id.items()}
    return "".join([id_printable[int(t)] for t in tensor])
