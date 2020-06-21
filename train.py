import os

import pytorch_lightning as pl

import dataloader
from config import *
from model import CharRNN

torch.autograd.set_detect_anomaly(True)  # shows errors when we messed up

(trl, tel, val), vocab = dataloader.load(FILE_NAME, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device=DEVICE, unique=True,
                                         splits=(0, 95, 99, 100))
params = {"model_name": "lstm", "hidden_size": HIDDEN_SIZE, "embedding_dim": EMBEDDING_DIM,
          "n_layers": N_LAYERS, "dropout": DROPOUT, "lr": LR}
net = CharRNN(vocab_size=len(vocab), **params).to(DEVICE)
if os.path.exists(CHECKPOINT_NAME):
    CharRNN.load_from_checkpoint(CHECKPOINT_NAME, vs=len(vocab), **params)
else:
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='avg_val_loss', min_delta=0.00, patience=3, mode="max")
    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), precision=16, gradient_clip_val=CLIP,
                         max_epochs=MAX_EPOCHS, progress_bar_refresh_rate=10, early_stop_callback=early_stop_callback)
    trainer.fit(net, train_dataloader=trl, val_dataloaders=val)
    trainer.save_checkpoint("net")

for method in "max", "rand", "softrand":
    print(net.predict("hello", 10, vocab, method=method))
