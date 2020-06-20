from string import printable

import torch
from torch.utils.data import DataLoader, Dataset


class ContStrDataSet(Dataset):
    def __init__(self, data, offset, end, sample_len, disjoint_samples):
        super(ContStrDataSet, self).__init__()
        self.data = data[offset:end]
        self.sample_len = sample_len
        self.disjoint_samples = disjoint_samples
        self.len = len(self.data) // self.sample_len if disjoint_samples else len(self.data) - sample_len

    def __len__(self):  # joined example: end=100, offset=10, sample_len=10 -> starting from 10 to 90
        return self.len

    def __getitem__(self, idx):
        start = idx * self.sample_len if self.disjoint_samples else idx
        return self.data[start: start + self.sample_len]


def load(file_name, device, splits=(0, 80, 90, 100), batch_size=100, seq_len=100, unique=True):
    data = open(file_name).read()
    assert set(data).issubset(printable), "Datset contains non-strings.printable-chars"
    printable_id = {s: i for i, s in enumerate(sorted(set(data)))}
    t = torch.tensor([printable_id[c] for c in data], dtype=torch.long, device=device)
    splits = [len(data) * i // 100 for i in splits]
    loaders = [DataLoader(ContStrDataSet(t, splits[i], splits[i + 1], seq_len, disjoint_samples=unique),
                          batch_size=batch_size, shuffle=not bool(splits[i])) for i in range(3)]
    return loaders, len(printable_id)
