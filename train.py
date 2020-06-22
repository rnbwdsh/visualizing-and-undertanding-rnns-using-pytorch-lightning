import pytorch_lightning as pl
import torch

import dataloader
from config import *
from net import CharRNN

torch.autograd.set_detect_anomaly(True)  # shows errors when we messed up

(trl, tel, val), vocab = dataloader.load(FILE_NAME, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device=DEVICE, unique=True,
                                         splits=(0, 95, 99, 100))

net = CharRNN(len(vocab), HIDDEN_SIZE, EMBEDDING_DIM, MODEL_NAME, DROPOUT, N_LAYERS, LR).to(DEVICE)
early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.00, patience=3, mode="max")
trainer = pl.Trainer(gpus=int(DEVICE == "cuda"), precision=PRECISION, gradient_clip_val=CLIP,
                     max_epochs=MAX_EPOCHS, progress_bar_refresh_rate=10, early_stop_callback=early_stop_callback)
trainer.fit(net, train_dataloader=trl, val_dataloaders=val)

for method in "max", "rand", "softrand":
    print(net.predict("Hello ", PREDICT_SEQ_LEN, vocab, method=method))
