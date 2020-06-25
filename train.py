import pytorch_lightning as pl

import dataloader
from config import *
from net.charrnn import CharRNN
from net.lightning import Lightning

(trl, tel, val), vocab = dataloader.load(FILE_PATH, DEVICE, SPLITS, BATCH_SIZE, SEQ_LEN, unique=True)

for MODEL_NAME in ["rnn", "gru", "lstm"]:
    for N_LAYERS in [1, 2, 3, 4]:
        for HIDDEN_SIZE in [32, 64, 128, 256, 512, 1024]:
            net = CharRNN(len(vocab), HIDDEN_SIZE, EMBEDDING_DIM, MODEL_NAME, DROPOUT, N_LAYERS, DEVICE)
            lightning = Lightning(net, LR)
            esc = pl.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.00, patience=3, mode="max")
            trainer = pl.Trainer(gpus=int(DEVICE == "cuda"), precision=PRECISION, gradient_clip_val=CLIP,
                                 max_epochs=MAX_EPOCHS, progress_bar_refresh_rate=10, early_stop_callback=esc,
                                 benchmark=True, fast_dev_run=False)
            trainer.fit(lightning, train_dataloader=trl, val_dataloaders=val)
            net.save_to_file()
