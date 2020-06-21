import torch.cuda

_FILE_NAMES = ["data/warandpeace.txt", "data/shakespeare.txt", "data/simple_example.txt"]
FILE_NAME = _FILE_NAMES[0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 100
PREDICT_SEQ_LEN = 150
HIDDEN_SIZE = 16
BATCH_SIZE = 100
N_LAYERS = 1
LR = 0.01
CLIP = 5
DROPOUT = 0.5
MAX_EPOCHS = 50
CHECKPOINT_NAME = "net"
EMBEDDING_DIM = 100
READ_LEN = 100000
