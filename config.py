import torch

# linux = https://cs.stanford.edu/people/karpathy/char-rnn/
_FILE_NAMES = ["warandpeace", "shakespeare", "linux"]
FILE_NAME = _FILE_NAMES[0]
FILE_PATH = f"data/{FILE_NAME}.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_LAYERS = 3
SEQ_LEN = 70
HIDDEN_SIZE = 32
BATCH_SIZE = 128
PREDICT_SEQ_LEN = SEQ_LEN
LR = 0.01
CLIP = 5
DROPOUT = 0.0
MAX_EPOCHS = 100
CHECKPOINT_NAME = "net"
EMBEDDING_DIM = 0  # 0 = one-hot encoding
MODEL_NAME = "lstm"
PRECISION = 32  # 16 if DEVICE == "cuda" else 32
SPLITS = (0, 90, 95, 100)
