import pytorch_lightning as pl
import torch

from dataloader import load_data
from model import CharRNN

torch.autograd.set_detect_anomaly(True)  # shows errors when we messed up

(trl, tel, val), nr_chars = load_data("data/warandpeace.txt", batch_size=100)  # "data/shakespeare.txt"
model = CharRNN(model="rnn", nr_chars=nr_chars, hidden_size=128, lr=0.01, embedding_dim=128)
trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), precision=32)
trainer.fit(model, train_dataloader=trl)  # val_dataloaders=val
