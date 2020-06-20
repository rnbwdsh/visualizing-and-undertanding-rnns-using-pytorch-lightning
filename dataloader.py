import torch
import string
from torch.utils.data import DataLoader, Dataset


class ContStrDataSet(Dataset):
    def __init__(self, data, offset, end, sample_len):
        super(ContStrDataSet, self).__init__()
        self.data = data[offset:end]
        self.sample_len = sample_len

    def __len__(self):  # example: end=100, offset=10, sample_len=10 -> starting from 10 to 90
        return len(self.data)//self.sample_len

    def __getitem__(self, idx):
        return self.data[idx*self.sample_len: idx*self.sample_len+self.sample_len]


def load_data(fn="data/warandpeace.txt", splits=(0, 80, 90, 100), batch_size=100, seq_len=100, device="cuda"):
    # d = open("data/shakespeare.txt").read()
    data = open(fn).read()
    assert set(data).issubset(string.printable), "Datset contains non-strings.printable-chars"
    printable_id = {s: i for i, s in enumerate(sorted(set(data)))}
    t = torch.LongTensor([printable_id[c] for c in data]).to(device)
    splits = [len(data)*i//100 for i in splits]
    loaders = []
    for i in range(3):
        ds = ContStrDataSet(t, splits[i], splits[i+1], seq_len)
        shuffle = not bool(splits[i])  # 1 for 0, 0 for all others
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle))
    return loaders, len(printable_id)
