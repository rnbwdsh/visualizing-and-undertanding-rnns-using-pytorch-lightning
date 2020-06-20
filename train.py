import torch
import pytorch_lightning as pl
from model import CharRNN
from dataloader import load_data

torch.autograd.set_detect_anomaly(True)


(trl, tel, val), nr_chars = load_data(batch_size=100)
model = CharRNN(model="rnn", nr_chars=nr_chars, hidden_size=128, lr=0.01, embedding_dim=128)
trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), precision=32)
trainer.fit(model, train_dataloader=trl)  # val_dataloaders=val
