DATA_PATH = './data/'
MODEL_PATH = './experiments/'

CUDA_DEVICE="cuda:0"


NUM_CLASS = 1
BATCH_SIZE = 100
TRAINING_EPOCHES = 100
NUM_PRETRAINING = 20


##################################
LEARNING_RATE = 0.001
DECAY_RATE = 0.1
STEP_SIZE = 10


##################################
#INPUT_DIM = 138
#HIDDEN_DIM = 138 
#INPUT_LEN = 7

INPUT_LEN = 24
INPUT_DIM_M1 = 25
INPUT_DIM_M2 = 4
EMBED_DIM = 20

HIDDEN_DIM = 32 