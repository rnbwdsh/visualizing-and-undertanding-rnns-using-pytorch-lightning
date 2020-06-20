import pytorch_lightning as pl
import torch

from dataloader import load
from model import CharRNN

torch.autograd.set_detect_anomaly(True)  # shows errors when we messed up

f = ["data/warandpeace.txt", "data/shakespeare.txt", "data/simple_example.txt"]
(trl, tel, val), nr_chars = load(f[0], batch_size=128, seq_len=128,
                                 device="cuda" if torch.cuda.is_available() else "cpu", unique=True)
model = CharRNN(model="lstm", nr_chars=nr_chars, hidden_size=512, embedding_dim=0, layers=2, lr=0.001)
trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), precision=32)  # , gradient_clip_val=1
trainer.fit(model, train_dataloader=trl, val_dataloaders=val)
