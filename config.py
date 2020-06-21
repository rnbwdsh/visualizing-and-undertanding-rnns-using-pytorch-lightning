import torch.cuda

_FILE_NAMES = ["data/warandpeace.txt", "data/shakespeare.txt", "data/simple_example.txt"]
FILE_NAME = _FILE_NAMES[0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 70
PREDICT_SEQ_LEN = SEQ_LEN
HIDDEN_SIZE = 512
BATCH_SIZE = 100
N_LAYERS = 2
LR = 0.01
CLIP = 5
DROPOUT = 0.0
MAX_EPOCHS = 1
CHECKPOINT_NAME = "net"
EMBEDDING_DIM = 0  # 0 = one-hot encoding
MODEL_NAME = "lstm"
