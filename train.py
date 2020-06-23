import pytorch_lightning as pl

import dataloader
from config import *
from net import CharRNN, Lightning

(trl, tel, val), vocab = dataloader.load(FILE_PATH, DEVICE, SPLITS, BATCH_SIZE, SEQ_LEN, unique=True)

for MODEL_NAME in ["gru", "lstm"]:
    for N_LAYERS in [1, 2, 3, 4]:
        for HIDDEN_SIZE in [32, 64, 128, 256, 512, 1024]:
            net = CharRNN(len(vocab), HIDDEN_SIZE, EMBEDDING_DIM, MODEL_NAME, DROPOUT, N_LAYERS, DEVICE)
            lightning = Lightning(net, LR)
            esc = pl.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.00, patience=3, mode="max")
            trainer = pl.Trainer(gpus=int(DEVICE == "cuda"), precision=PRECISION, gradient_clip_val=CLIP,
                                 max_epochs=MAX_EPOCHS, progress_bar_refresh_rate=10, early_stop_callback=esc,
                                 benchmark=True,
                                 fast_dev_run=False)  # auto_lr_find=True, auto_scale_batch_size=True needs hparams
            trainer.fit(lightning, train_dataloader=trl, val_dataloaders=val)
            with open(f"models/{MODEL_NAME}-{N_LAYERS}-{HIDDEN_SIZE}-{FILE_NAME}.pkl", "wb") as out:
                torch.save(net.state_dict(), out)

# for method in "max", "rand", "softrand":
#    print(net.predict("Hello ", PREDICT_SEQ_LEN, vocab, method=method))
