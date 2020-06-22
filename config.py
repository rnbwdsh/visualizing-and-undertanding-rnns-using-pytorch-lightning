_FILE_NAMES = ["data/warandpeace.txt", "data/shakespeare.txt", "data/simple_example.txt"]
FILE_NAME = _FILE_NAMES[0]
DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
N_LAYERS = 2
SEQ_LEN = 3
PREDICT_SEQ_LEN = SEQ_LEN
HIDDEN_SIZE = 1
BATCH_SIZE = 1
LR = 0.01
CLIP = 5
DROPOUT = 0.0
MAX_EPOCHS = 1
CHECKPOINT_NAME = "net"
EMBEDDING_DIM = 0  # 0 = one-hot encoding
MODEL_NAME = "gru"
PRECISION = 16 if DEVICE == "cuda" else 32
